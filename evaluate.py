import argparse
import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

CONFIG = {
    "seed": 3333,
    "eval_config": {
        "large_data_threshold": 3000,
        "large_k": 30,
        "small_k": 10,
        "resolution_search_range": np.concatenate([np.arange(0.001, 2.5, 0.005)]),
    },
}


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def safe_silhouette_score(X, labels, metric="cosine"):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan
    try:
        return silhouette_score(X, labels, metric=metric)
    except Exception:
        return np.nan


def resolve_adata_path(adata_path=None, h5ad_dir="h5ad", dataset_name=None):
    if adata_path is None:
        if dataset_name is None:
            raise ValueError("Either adata_path or dataset_name must be provided.")
        adata_path = os.path.join(h5ad_dir, f"{dataset_name}.h5ad")

    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    return adata_path


def infer_dataset_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0].replace("_evaluated", "")


def get_dataset_name(adata, adata_path, dataset_name=None):
    if dataset_name is not None:
        return dataset_name
    if "dataset_name" in adata.uns:
        return str(adata.uns["dataset_name"])
    return infer_dataset_name_from_path(adata_path)


def select_embedding_key(adata, requested_key=None):
    if requested_key is not None:
        if requested_key not in adata.obsm:
            raise KeyError(f"Embedding key not found in adata.obsm: {requested_key}")
        return requested_key

    if "embedding_key" in adata.uns:
        key = str(adata.uns["embedding_key"])
        if key in adata.obsm:
            return key

    for key in ["X_masked_clustering", "X_emb", "X_embedding"]:
        if key in adata.obsm:
            return key

    raise KeyError("No embedding key found. Pass --embedding-key or store embeddings in adata.obsm.")


def get_true_labels(adata, label_key=None, cell_type_key="cell_type"):
    if label_key is not None:
        if label_key not in adata.obs:
            raise KeyError(f"Label key not found in adata.obs: {label_key}")
        values = np.asarray(adata.obs[label_key])
        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.int64), label_key
        return LabelEncoder().fit_transform(values.astype(str)), label_key

    if "label_encoded" in adata.obs:
        return np.asarray(adata.obs["label_encoded"]).astype(np.int64), "label_encoded"

    if cell_type_key in adata.obs:
        values = np.asarray(adata.obs[cell_type_key].astype(str))
        return LabelEncoder().fit_transform(values), cell_type_key

    return None, None


def h5ad_metrics_table(result_rows):
    if not result_rows:
        return {}
    df = pd.DataFrame(result_rows)
    return {col: df[col].to_numpy() for col in df.columns}


def evaluate_anndata(adata, cfg=None, dataset_name=None, embedding_key=None, label_key=None, cell_type_key="cell_type", output_dir="outputs"):
    if cfg is None:
        cfg = CONFIG

    os.makedirs(output_dir, exist_ok=True)

    embedding_key = select_embedding_key(adata, embedding_key)
    embeddings = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    eval_cfg = cfg["eval_config"]
    num_cells = embeddings.shape[0]

    if num_cells > eval_cfg["large_data_threshold"]:
        eval_neighbors = eval_cfg["large_k"]
    else:
        eval_neighbors = eval_cfg["small_k"]

    true_labels, used_label_key = get_true_labels(adata, label_key=label_key, cell_type_key=cell_type_key)
    has_labels = true_labels is not None

    if has_labels and cell_type_key not in adata.obs:
        adata.obs[cell_type_key] = np.asarray(true_labels).astype(str)
        adata.obs[cell_type_key] = adata.obs[cell_type_key].astype("category")

    neighbors_key = "masked_clustering_neighbors"
    best_cluster_key = "leiden_best"
    temp_cluster_key = "leiden_tmp"

    print(f"Loaded AnnData with shape: {adata.shape}")
    print(f"Using embedding key: {embedding_key}")
    print(f"Evaluation n_neighbors={eval_neighbors} for {num_cells} cells.")

    print("Computing neighbors and UMAP.")
    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=eval_neighbors, metric="cosine", key_added=neighbors_key)
    sc.tl.umap(adata, neighbors_key=neighbors_key, random_state=cfg["seed"])

    best_metrics = [0.0, 0.0, 0.0, 0.0, np.nan]
    best_res = 0.0
    best_n_clusters = 0
    best_score = -np.inf
    best_labels = None
    result_rows = []

    resolutions = eval_cfg["resolution_search_range"]

    print("Searching for the best clustering resolution.")
    print("-" * 125)
    if has_labels:
        print(f"{'Resolution':<10} | {'ARI':<8} | {'NMI':<8} | {'AMI':<8} | {'ACC':<8} | {'SIL':<8} | {'Clusters':<8}")
    else:
        print(f"{'Resolution':<10} | {'SIL':<8} | {'Clusters':<8}")
    print("-" * 125)

    for res in resolutions:
        try:
            sc.tl.leiden(
                adata,
                key_added=temp_cluster_key,
                resolution=float(res),
                random_state=cfg["seed"],
                neighbors_key=neighbors_key,
            )
            labels_p = adata.obs[temp_cluster_key].astype(int).values
            n_clusters = len(np.unique(labels_p))
            sil = safe_silhouette_score(embeddings, labels_p, metric="cosine")

            if has_labels:
                ari = adjusted_rand_score(true_labels, labels_p)
                nmi = normalized_mutual_info_score(true_labels, labels_p)
                ami = adjusted_mutual_info_score(true_labels, labels_p)
                acc = cluster_acc(true_labels, labels_p)
                print(f"{res:<10.5f} | {ari:<8.4f} | {nmi:<8.4f} | {ami:<8.4f} | {acc:<8.4f} | {sil:<8.4f} | {n_clusters:<8}")
                result_rows.append(
                    {
                        "resolution": float(res),
                        "ari": float(ari),
                        "nmi": float(nmi),
                        "ami": float(ami),
                        "acc": float(acc),
                        "sil": float(sil) if not np.isnan(sil) else np.nan,
                        "n_clusters": int(n_clusters),
                    }
                )

                score = ari
                if score > best_score:
                    best_score = score
                    best_metrics = [ari, nmi, ami, acc, sil]
                    best_res = float(res)
                    best_n_clusters = int(n_clusters)
                    best_labels = adata.obs[temp_cluster_key].copy()
            else:
                print(f"{res:<10.5f} | {sil:<8.4f} | {n_clusters:<8}")
                result_rows.append(
                    {
                        "resolution": float(res),
                        "sil": float(sil) if not np.isnan(sil) else np.nan,
                        "n_clusters": int(n_clusters),
                    }
                )

                score = sil if not np.isnan(sil) else -np.inf
                if score > best_score:
                    best_score = score
                    best_res = float(res)
                    best_n_clusters = int(n_clusters)
                    best_metrics[4] = sil
                    best_labels = adata.obs[temp_cluster_key].copy()
        except Exception as err:
            print(f"Skipping resolution {float(res):.5f}: {err}")

    if temp_cluster_key in adata.obs:
        del adata.obs[temp_cluster_key]

    if best_labels is not None:
        adata.obs[best_cluster_key] = best_labels.astype("category")

    print("-" * 125)
    if has_labels:
        print(
            f"Best metrics: ARI={best_metrics[0]:.5f}, NMI={best_metrics[1]:.5f}, AMI={best_metrics[2]:.5f}, ACC={best_metrics[3]:.5f}, SIL={best_metrics[4]:.5f}, Number of clusters={best_n_clusters}"
        )
    else:
        print(f"Best metrics: SIL={best_metrics[4]:.5f}, Number of clusters={best_n_clusters}")

    suffix = f"_{dataset_name}" if dataset_name else ""
    umap_path = os.path.join(output_dir, f"umap_result{suffix}.png")
    metrics_path = os.path.join(output_dir, f"metrics{suffix}.csv")

    print("Generating UMAP plot.")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if has_labels and cell_type_key in adata.obs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sc.pl.umap(adata, color=[cell_type_key], title=f"Ground Truth (ARI={best_metrics[0]:.5f})", show=False, ax=ax1)
        if best_cluster_key in adata.obs:
            sc.pl.umap(
                adata,
                color=[best_cluster_key],
                title=f"Predicted (Res={best_res:.2f}, Clusters={best_n_clusters}, SIL={best_metrics[4]:.5f})",
                show=False,
                ax=ax2,
            )
        plt.tight_layout()
        plt.savefig(umap_path)
        plt.close()
    else:
        if best_cluster_key in adata.obs:
            sc.pl.umap(
                adata,
                color=[best_cluster_key],
                title=f"Predicted Clustering (Res={best_res:.2f}, Clusters={best_n_clusters}, SIL={best_metrics[4]:.5f})",
                show=False,
            )
            plt.tight_layout()
            plt.savefig(umap_path)
            plt.close()

    metrics_df = pd.DataFrame(result_rows)
    metrics_df.to_csv(metrics_path, index=False)

    best_metrics_dict = {
        "ari": float(best_metrics[0]) if has_labels else np.nan,
        "nmi": float(best_metrics[1]) if has_labels else np.nan,
        "ami": float(best_metrics[2]) if has_labels else np.nan,
        "acc": float(best_metrics[3]) if has_labels else np.nan,
        "sil": float(best_metrics[4]) if not np.isnan(best_metrics[4]) else np.nan,
    }

    adata.uns["masked_clustering_eval"] = {
        "stage": "evaluated",
        "dataset_name": str(dataset_name) if dataset_name else "unknown",
        "embedding_key": str(embedding_key),
        "neighbors_key": neighbors_key,
        "cluster_key": best_cluster_key,
        "label_key": str(used_label_key) if used_label_key is not None else "None",
        "best_resolution": float(best_res),
        "best_n_clusters": int(best_n_clusters),
        "selected_by": "ari" if has_labels else "sil",
        "best_metrics": best_metrics_dict,
        "metrics_table": h5ad_metrics_table(result_rows),
        "umap_path": umap_path,
        "metrics_path": metrics_path,
    }

    print(f"Saved UMAP plot to: {umap_path}")
    print(f"Saved metrics table to: {metrics_path}")

    return adata, best_metrics, best_n_clusters, umap_path, metrics_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved AnnData object with Leiden clustering.")
    parser.add_argument("--adata-path", type=str, default=None, help="Direct path to a saved .h5ad file.")
    parser.add_argument("--h5ad-dir", type=str, default="embeddings", help="Directory containing saved .h5ad files.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name used to locate h5ad/<dataset_name>.h5ad.")
    parser.add_argument("--embedding-key", type=str, default=None, help="Embedding key in adata.obsm. Defaults to the key saved by train.py.")
    parser.add_argument("--label-key", type=str, default=None, help="Ground-truth label key in adata.obs.")
    parser.add_argument("--cell-type-key", type=str, default="cell_type", help="Cell type key used for the ground-truth UMAP panel.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots and metric tables.")
    parser.add_argument("--save-path", type=str, default=None, help="Path for the evaluated .h5ad output file.")
    parser.add_argument("--overwrite-input", action="store_true", help="Overwrite the input .h5ad with evaluated results.")
    parser.add_argument("--seed", type=int, default=3333, help="Random seed.")
    parser.add_argument("--large-data-threshold", type=int, default=3000, help="Cell-count threshold for using large_k.")
    parser.add_argument("--large-k", type=int, default=30, help="n_neighbors for large datasets.")
    parser.add_argument("--small-k", type=int, default=10, help="n_neighbors for small datasets.")
    parser.add_argument("--resolution-start", type=float, default=0.001, help="Resolution search start.")
    parser.add_argument("--resolution-stop", type=float, default=2.5, help="Resolution search stop.")
    parser.add_argument("--resolution-step", type=float, default=0.005, help="Resolution search step.")
    return parser.parse_args()


def build_runtime_config(args):
    cfg = {
        "seed": args.seed,
        "eval_config": {
            "large_data_threshold": args.large_data_threshold,
            "large_k": args.large_k,
            "small_k": args.small_k,
            "resolution_search_range": np.arange(args.resolution_start, args.resolution_stop, args.resolution_step),
        },
    }
    return cfg


def resolve_save_path(args, input_path, dataset_name):
    if args.overwrite_input:
        return input_path
    if args.save_path is not None:
        return args.save_path
    base_dir = os.path.dirname(input_path) or "."
    return os.path.join(base_dir, f"{dataset_name}_evaluated.h5ad")


if __name__ == "__main__":
    parsed_args = parse_args()
    runtime_cfg = build_runtime_config(parsed_args)
    input_path = resolve_adata_path(
        adata_path=parsed_args.adata_path,
        h5ad_dir=parsed_args.h5ad_dir,
        dataset_name=parsed_args.dataset_name,
    )

    loaded_adata = ad.read_h5ad(input_path)
    loaded_dataset_name = get_dataset_name(loaded_adata, input_path, parsed_args.dataset_name)

    evaluated_adata, metrics, n_clusters, _, _ = evaluate_anndata(
        loaded_adata,
        cfg=runtime_cfg,
        dataset_name=loaded_dataset_name,
        embedding_key=parsed_args.embedding_key,
        label_key=parsed_args.label_key,
        cell_type_key=parsed_args.cell_type_key,
        output_dir=parsed_args.output_dir,
    )

    output_h5ad_path = resolve_save_path(parsed_args, input_path, loaded_dataset_name)
    evaluated_adata.write_h5ad(output_h5ad_path, compression="gzip")
    print(f"Saved evaluated AnnData to: {output_h5ad_path}")

    print("Final Results:")
    if not np.isnan(metrics[0]):
        print(f"ARI: {metrics[0]:.5f}")
        print(f"NMI: {metrics[1]:.5f}")
        print(f"AMI: {metrics[2]:.5f}")
        print(f"ACC: {metrics[3]:.5f}")
    print(f"SIL: {metrics[4]:.5f}")
    print(f"Number of predicted clusters: {n_clusters}")
    print(f"Evaluated AnnData file: {output_h5ad_path}")
