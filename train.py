import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loader import RNADataset
from mask_sampler import MaskSampler

CONFIG = {
    "rna_path": None,
    "data_dir": "data",
    "data_file": "data.h5ad",
    "seed": 3333,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "gpu": 0,
    "h5ad_dir": "h5ad",
    "embedding_key": "X_masked_clustering",
    "data_config": {
        "n_hvgs": 3000,
        "min_counts": 3,
        "min_genes": 200,
        "target_sum": 1e4,
        "max_value": 10,
        "n_pcs": 50,
    },
    "graph_config": {
        "k_base": 30,
        "min_shared": 10,
        "h_topk": 15,
        "lap_topk": 20,
    },
    "module_config": {
        "k_genes": 25,
        "k_pb_cells": 30,
        "res_pb_cells": 2.0,
        "res_gene_modules": 5.0,
    },
    "model_config": {
        "emb_size": 64,
        "hid_dim": 256,
        "proj_hidden": 256,
        "proj_out": 64,
    },
    "training_config": {
        "batch_size": 64,
        "epochs": 250,
        "curriculum_epochs": 50,
        "lambda_rec": 1.0,
        "lambda_rep": 3.0,
        "lambda_lap": 1.0,
        "lr": 1e-3,
        "temp_start": 0.1,
        "temp_end": 0.05,
        "weight_decay": 1e-4,
    },
    "masking_config": {
        "initial_mask_frac": 0.06,
        "final_mask_frac": 0.50,
        "module_frac_start": 0.05,
        "module_frac_end": 0.25,
        "module_mask_frac": 0.5,
    },
}


def infer_dataset_name(rna_path):
    path = os.path.normpath(rna_path)
    file_stem = os.path.splitext(os.path.basename(path))[0]
    if file_stem.lower() in {"data", "matrix", "counts", "expression", "exprs"}:
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            return parent
    return file_stem


def get_dataset_name(cfg):
    if cfg.get("dataset_name"):
        return cfg["dataset_name"]
    if cfg.get("rna_path"):
        return infer_dataset_name(cfg["rna_path"])
    raise ValueError("Dataset name could not be inferred. Provide --dataset-name or --rna-path.")


def get_project_dir():
    return Path(__file__).resolve().parent


def resolve_path_relative_to_project(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return get_project_dir() / path


def resolve_dataset_rna_path(dataset_name, data_dir="data", data_file="data.h5ad"):
    if not dataset_name:
        raise ValueError("dataset_name is required when --rna-path is not provided.")

    base_dir = resolve_path_relative_to_project(data_dir)
    candidate = base_dir / dataset_name / data_file
    if candidate.exists():
        return str(candidate)

    fallback_candidates = [
        base_dir / dataset_name / "data.h5ad",
        base_dir / dataset_name / "data.h5",
    ]
    checked_paths = [candidate]

    for fallback in fallback_candidates:
        if fallback not in checked_paths:
            checked_paths.append(fallback)
            if fallback.exists():
                return str(fallback)

    checked_text = "\n".join(str(path) for path in checked_paths)
    raise FileNotFoundError(f"Dataset file was not found. Checked paths:\n{checked_text}")


def to_h5ad_safe(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_h5ad_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [to_h5ad_safe(v) for v in value]
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_binary_mutual_topk(W, k=15, fallback_for_isolated=True):
    n = W.shape[0]
    A = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        nz = np.flatnonzero(W[i] > 0)
        if len(nz) == 0:
            continue

        if len(nz) <= k:
            keep = nz
        else:
            vals = W[i, nz]
            idx = np.argpartition(vals, -k)[-k:]
            keep = nz[idx]

        A[i, keep] = 1.0

    A = np.minimum(A, A.T)
    np.fill_diagonal(A, 0.0)

    if fallback_for_isolated:
        deg = A.sum(axis=1)
        isolated_nodes = np.where(deg == 0)[0]

        for i in isolated_nodes:
            nz = np.flatnonzero(W[i] > 0)
            if len(nz) == 0:
                continue
            j = nz[np.argmax(W[i, nz])]
            A[i, j] = 1.0
            A[j, i] = 1.0

    np.fill_diagonal(A, 0.0)
    return A


def build_weighted_topk_graph(W, k=20):
    n = W.shape[0]
    A = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        nz = np.flatnonzero(W[i] > 0)
        if len(nz) == 0:
            continue

        if len(nz) <= k:
            keep = nz
        else:
            vals = W[i, nz]
            idx = np.argpartition(vals, -k)[-k:]
            keep = nz[idx]

        A[i, keep] = W[i, keep]

    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.skip_weight = nn.Parameter(torch.tensor(0.3))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, X, H):
        H = H.to(X.device)

        De = H.sum(dim=0)
        Dv = H.sum(dim=1)

        De_inv = De.pow(-1.0)
        Dv_inv_sqrt = Dv.pow(-0.5)

        De_inv.masked_fill_(torch.isinf(De_inv) | torch.isnan(De_inv), 0.0)
        Dv_inv_sqrt.masked_fill_(torch.isinf(Dv_inv_sqrt) | torch.isnan(Dv_inv_sqrt), 0.0)

        H_De = H * De_inv.unsqueeze(0)
        adj = torch.matmul(H_De, H.T)

        P = adj * Dv_inv_sqrt.unsqueeze(1) * Dv_inv_sqrt.unsqueeze(0)

        base_transform = X @ self.theta
        out = P @ base_transform
        out = self.skip_weight * out + (1.0 - self.skip_weight) * base_transform

        if self.bias is not None:
            out = out + self.bias

        return out


class HypergraphAE(nn.Module):
    def __init__(self, in_dim, hid_dim, emb_dim):
        super().__init__()
        self.hg1 = HypergraphConv(in_dim, hid_dim)
        self.hg2 = HypergraphConv(hid_dim, emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
        )

    def forward(self, X, H):
        h = F.relu(self.hg1(X, H))
        z = self.hg2(h, H)
        Xrec = self.decoder(F.relu(z))
        return z, Xrec


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = (~torch.eye(2 * B, dtype=torch.bool, device=z.device)).float()
        exp_sim = torch.exp(sim) * mask
        denom = exp_sim.sum(dim=1)
        pos = torch.exp((z1 * z2).sum(dim=1) / self.temperature)
        loss = -torch.log(pos / denom[:B]) - torch.log(pos / denom[B:])
        return loss.mean()


class HypergraphAEWrapper(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_size, proj_hidden, proj_out):
        super().__init__()
        self.hg_model = HypergraphAE(in_dim=input_dim, hid_dim=hid_dim, emb_dim=emb_size)
        self.proj = ProjectionHead(emb_size, hidden=proj_hidden, out_dim=proj_out)

    def forward(self, x, H):
        if H is None:
            raise ValueError("H must be provided to HypergraphAEWrapper.forward.")
        z, recon = self.hg_model(x, H)
        proj = self.proj(z)
        return recon, z, proj


def recon_loss_masked(x_hat, x_true, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_hat.device, requires_grad=True)
    return F.mse_loss(x_hat[mask], x_true[mask])


def laplacian_loss(embeddings, W):
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
    loss = 0.5 * (W.unsqueeze(2) * (diff ** 2)).sum() / embeddings.size(0)
    return loss


def extract_embeddings(model, rna_data, cfg):
    train_cfg = cfg["training_config"]
    device = cfg["device"]
    rna_loader = DataLoader(rna_data, batch_size=train_cfg["batch_size"], shuffle=False, drop_last=False)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i, Xb in enumerate(rna_loader):
            Xb = Xb.to(device)
            idx_start = i * train_cfg["batch_size"]
            idx_end = idx_start + Xb.shape[0]
            H_batch = model.H_global[idx_start:idx_end, :]
            _, z, _ = model(Xb, H=H_batch)
            z_norm = F.normalize(z, p=2, dim=1)
            embeddings.append(z_norm)

    return torch.cat(embeddings, dim=0).cpu().numpy()


def save_training_h5ad(embeddings, rna_data, cfg, loss_history):
    dataset_name = get_dataset_name(cfg)
    h5ad_dir = cfg.get("h5ad_dir", "h5ad")
    embedding_key = cfg.get("embedding_key", "X_masked_clustering")
    os.makedirs(h5ad_dir, exist_ok=True)
    save_path = os.path.join(h5ad_dir, f"{dataset_name}.h5ad")

    adata = rna_data.rna_data.copy()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if adata.n_obs != embeddings.shape[0]:
        raise ValueError(f"Embedding row count {embeddings.shape[0]} does not match AnnData n_obs {adata.n_obs}.")

    adata.obsm[embedding_key] = embeddings
    if embedding_key != "X_emb":
        adata.obsm["X_emb"] = embeddings

    if hasattr(rna_data, "X") and rna_data.X.shape == adata.shape:
        adata.layers["scaled_hvg"] = np.asarray(rna_data.X, dtype=np.float32)

    if getattr(rna_data, "labels", None) is not None:
        adata.obs["label_encoded"] = np.asarray(rna_data.labels, dtype=np.int64)

    if getattr(rna_data, "cell_types", None) is not None:
        adata.obs["cell_type"] = np.asarray(rna_data.cell_types.astype(str).values)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    adata.uns["dataset_name"] = dataset_name
    adata.uns["rna_path"] = cfg["rna_path"]
    adata.uns["embedding_key"] = embedding_key
    adata.uns["masked_clustering_train"] = {
        "stage": "trained",
        "config": to_h5ad_safe(cfg),
        "loss_history": {k: np.asarray(v, dtype=np.float32 if k != "epoch" else np.int64) for k, v in loss_history.items()},
    }

    adata.write_h5ad(save_path, compression="gzip")
    print(f"Saved trained AnnData to: {save_path}")
    return save_path


def train(cfg):
    device = cfg["device"]
    seed = cfg["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(cfg["gpu"])
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    rna_data = RNADataset(cfg["rna_path"], cfg["data_config"]["n_hvgs"], cfg=cfg)

    cfg["num_genes"] = rna_data.X.shape[1]
    cfg["num_cells"] = rna_data.X.shape[0]
    W = rna_data.W

    if sp.issparse(W):
        W = W.toarray()

    print("Applying max symmetrization to the adjacency matrix for Laplacian loss.")
    W_lap_full = np.maximum(W, W.T).astype(np.float32)
    lap_topk = cfg["graph_config"].get("lap_topk", 20)
    W_lap = build_weighted_topk_graph(W_lap_full, k=lap_topk)
    row_sums = W_lap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_lap = W_lap / row_sums
    rna_data.W = W_lap

    mask_cfg = cfg["masking_config"]
    train_cfg = cfg["training_config"]
    model_cfg = cfg["model_config"]

    h_topk = cfg["graph_config"].get("h_topk", 15)
    H_bin = build_binary_mutual_topk(W_lap_full, k=h_topk, fallback_for_isolated=True)

    avg_h_degree = H_bin.sum(axis=1).mean()
    isolated = np.sum(H_bin.sum(axis=1) == 0)
    print("Loading global SNN incidence matrix.")
    H_matrix = torch.tensor(H_bin, dtype=torch.float32)
    I = torch.eye(cfg["num_cells"], dtype=torch.float32)
    H_global = H_matrix + I
    print(f"Global H matrix shape: {H_global.shape}")

    model = HypergraphAEWrapper(
        input_dim=cfg["num_genes"],
        hid_dim=model_cfg["hid_dim"],
        emb_size=model_cfg["emb_size"],
        proj_hidden=model_cfg["proj_hidden"],
        proj_out=model_cfg["proj_out"],
    ).to(device)

    model.register_buffer("H_global", H_global.to(device))

    sampler = MaskSampler(
        num_genes=cfg["num_genes"],
        modules=rna_data.modules,
        initial_mask_frac=mask_cfg["initial_mask_frac"],
        final_mask_frac=mask_cfg["final_mask_frac"],
        module_frac_start=mask_cfg["module_frac_start"],
        module_frac_end=mask_cfg["module_frac_end"],
        module_mask_frac=mask_cfg["module_mask_frac"],
        curriculum_epochs=train_cfg["curriculum_epochs"],
        seed=seed,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=train_cfg["epochs"],
        eta_min=1e-5,
    )

    criterion = InfoNCE(temperature=train_cfg["temp_start"])

    loss_history = {
        "epoch": [],
        "total_loss": [],
        "rec_loss": [],
        "con_loss": [],
        "lap_loss": [],
        "mask_frac": [],
    }

    print("Starting training.")
    print(f"{'Epoch':^5} | {'Loss':^8} | {'Rec':^8} | {'Con':^8} | {'Lap':^8} | {'Temp':^6}")
    print("-" * 80)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        sampler.update_schedule(epoch)

        if epoch <= train_cfg["curriculum_epochs"]:
            alpha = (epoch - 1) / max(1, train_cfg["curriculum_epochs"] - 1)
            current_temp = train_cfg["temp_start"] + alpha * (train_cfg["temp_end"] - train_cfg["temp_start"])
        else:
            current_temp = train_cfg["temp_end"]

        criterion.temperature = current_temp

        if device.type == "cuda" and epoch % 5 == 0:
            torch.cuda.empty_cache()

        rindexs = np.arange(len(rna_data))
        np.random.shuffle(rindexs)
        rna_loader = DataLoader(rna_data[rindexs], batch_size=train_cfg["batch_size"], shuffle=False, drop_last=True)

        x_perm = rna_data.X.copy()
        for j in range(x_perm.shape[1]):
            np.random.shuffle(x_perm[:, j])
        x_perm = torch.tensor(x_perm, dtype=torch.float32)

        We = torch.tensor(W_lap[rindexs][:, rindexs], dtype=torch.float32)

        total_loss = 0.0
        total_rec = 0.0
        total_con = 0.0
        total_lap = 0.0
        iters = 0

        for i, Xb in enumerate(rna_loader):
            Xb = Xb.to(device)
            B = Xb.shape[0]
            Xm = x_perm[i * B : (i + 1) * B, :].to(device)
            Wb = We[i * B : (i + 1) * B, i * B : (i + 1) * B].to(device)

            batch_indices = rindexs[i * B : (i + 1) * B]
            H_batch = model.H_global[batch_indices, :]

            mask1, mask2 = sampler.sample_paired_views(B)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)

            view1 = torch.where(mask1, Xm, Xb)
            view2 = torch.where(mask2, Xm, Xb)

            xhat1, z1, proj1 = model(view1, H=H_batch)
            xhat2, z2, proj2 = model(view2, H=H_batch)

            loss_rec = 0.5 * (recon_loss_masked(xhat1, Xb, mask1) + recon_loss_masked(xhat2, Xb, mask2))
            loss_contr = criterion(proj1, proj2)
            loss_lap = 0.5 * (laplacian_loss(z1, Wb) + laplacian_loss(z2, Wb))

            loss = (
                train_cfg["lambda_rec"] * loss_rec
                + train_cfg["lambda_rep"] * loss_contr
                + train_cfg["lambda_lap"] * loss_lap
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()
            total_rec += loss_rec.item()
            total_con += loss_contr.item()
            total_lap += loss_lap.item()
            iters += 1

        scheduler.step()

        avg_loss = total_loss / max(1, iters)
        avg_rec = total_rec / max(1, iters)
        avg_con = total_con / max(1, iters)
        avg_lap = total_lap / max(1, iters)

        loss_history["epoch"].append(epoch)
        loss_history["total_loss"].append(avg_loss)
        loss_history["rec_loss"].append(avg_rec)
        loss_history["con_loss"].append(avg_con)
        loss_history["lap_loss"].append(avg_lap)
        loss_history["mask_frac"].append(sampler.mask_frac)

        if epoch % 10 == 0 or epoch == train_cfg["epochs"]:
            print(f"{epoch:03d}   | {avg_loss:.4f}   | {avg_rec:.4f}   | {avg_con:.4f}   | {avg_lap:.4f}   | {current_temp:.3f}")

    print("Training completed. Extracting embeddings.")
    embeddings = extract_embeddings(model, rna_data, cfg)
    h5ad_path = save_training_h5ad(embeddings, rna_data, cfg, loss_history)

    return model, sampler, rna_data, loss_history, h5ad_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the masked clustering model and save a full AnnData object.")
    parser.add_argument("--rna-path", type=str, default=None, help="Direct path to a .h5 or .h5ad RNA dataset. If provided, it overrides --dataset-name lookup.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset folder name under --data-dir. The default lookup is data/<dataset_name>/data.h5ad.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset folders. Relative paths are resolved from the train.py directory.")
    parser.add_argument("--data-file", type=str, default="data.h5ad", help="File name inside each dataset folder.")
    parser.add_argument("--h5ad-dir", type=str, default="embeddings", help="Directory for saved AnnData files.")
    parser.add_argument("--embedding-key", type=str, default="X_masked_clustering", help="Key used in adata.obsm for saved embeddings.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index.")
    parser.add_argument("--seed", type=int, default=3333, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    return parser.parse_args()


def build_runtime_config(args):
    cfg = copy.deepcopy(CONFIG)
    cfg["data_dir"] = args.data_dir
    cfg["data_file"] = args.data_file

    if args.dataset_name is not None:
        cfg["dataset_name"] = args.dataset_name

    if args.rna_path is not None:
        rna_path = resolve_path_relative_to_project(args.rna_path)
        cfg["rna_path"] = str(rna_path)
        if args.dataset_name is None:
            cfg["dataset_name"] = infer_dataset_name(str(rna_path))
    else:
        dataset_name = cfg.get("dataset_name")
        if dataset_name is None:
            raise ValueError("Provide --dataset-name to use data/<dataset_name>/data.h5ad, or provide --rna-path directly.")
        cfg["rna_path"] = resolve_dataset_rna_path(dataset_name, data_dir=args.data_dir, data_file=args.data_file)

    cfg["h5ad_dir"] = args.h5ad_dir
    cfg["embedding_key"] = args.embedding_key
    if args.gpu is not None:
        cfg["gpu"] = args.gpu
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["training_config"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training_config"]["batch_size"] = args.batch_size
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset name: {get_dataset_name(cfg)}")
    print(f"RNA path: {cfg['rna_path']}")
    print(f"AnnData output directory: {cfg['h5ad_dir']}")
    return cfg


if __name__ == "__main__":
    args = parse_args()
    runtime_cfg = build_runtime_config(args)
    _, _, _, _, saved_h5ad_path = train(runtime_cfg)
    print("Final output:")
    print(f"AnnData file: {saved_h5ad_path}")
