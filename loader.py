import random

import anndata as ad
import h5py
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.stats as ss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from torch.utils.data import Dataset


def decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = np.vectorize(decode)(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    output = {}
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        output[key] = value
    return output


def read_h5_to_anndata(filename):
    print(f"Reading H5 file: {filename}")
    with h5py.File(filename, "r") as h5_file:
        if isinstance(h5_file["obs"], h5py.Group):
            obs_dict = dict_from_group(h5_file["obs"])
            obs = pd.DataFrame(obs_dict)
            if "obs_names" in h5_file:
                obs.index = np.vectorize(decode)(h5_file["obs_names"][...])
        else:
            print("Detected obs as a dataset. Converting to a DataFrame.")
            raw_data = h5_file["obs"][...]
            if raw_data.dtype.names:
                obs = pd.DataFrame(raw_data)
            else:
                obs = pd.DataFrame(raw_data)
                for col in obs.columns:
                    if len(obs) > 0 and isinstance(obs[col].iloc[0], bytes):
                        try:
                            obs[col] = obs[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
                        except Exception:
                            pass

            if "obs_names" in h5_file:
                obs.index = np.vectorize(decode)(h5_file["obs_names"][...])
            elif "barcodes" in h5_file:
                obs.index = np.vectorize(decode)(h5_file["barcodes"][...])

        if isinstance(h5_file["var"], h5py.Group):
            var_dict = dict_from_group(h5_file["var"])
            var = pd.DataFrame(var_dict)
            if "var_names" in h5_file:
                var.index = np.vectorize(decode)(h5_file["var_names"][...])
        else:
            print("Detected var as a dataset. Converting to a DataFrame.")
            raw_data = h5_file["var"][...]
            if raw_data.dtype.names:
                var = pd.DataFrame(raw_data)
            else:
                var = pd.DataFrame(raw_data)
                for col in var.columns:
                    if len(var) > 0 and isinstance(var[col].iloc[0], bytes):
                        try:
                            var[col] = var[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
                        except Exception:
                            pass

            if "var_names" in h5_file:
                var.index = np.vectorize(decode)(h5_file["var_names"][...])
            elif "genes" in h5_file:
                var.index = np.vectorize(decode)(h5_file["genes"][...])
            elif "gene_names" in h5_file:
                var.index = np.vectorize(decode)(h5_file["gene_names"][...])
            elif 0 in var.columns:
                var.index = var[0].astype(str)

        if "X" in h5_file:
            if isinstance(h5_file["X"], h5py.Group):
                data = h5_file["X"]["data"][...]
                indices = h5_file["X"]["indices"][...]
                indptr = h5_file["X"]["indptr"][...]
                shape = h5_file["X"].attrs.get("shape", (len(obs), len(var)))
                if "shape" in h5_file["X"]:
                    shape = h5_file["X"]["shape"][...]
                mat = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                mat = h5_file["X"][...].astype(np.float32)
        elif "exprs" in h5_file:
            exprs = h5_file["exprs"]
            if isinstance(exprs, h5py.Group):
                mat = sp.csr_matrix(
                    (exprs["data"][...], exprs["indices"][...], exprs["indptr"][...]),
                    shape=exprs["shape"][...],
                )
            else:
                mat = exprs[...].astype(np.float32)
        else:
            print("Warning: X or exprs was not found. Initializing a zero matrix.")
            mat = sp.csr_matrix((len(obs), len(var)), dtype=np.float32)

        uns = dict_from_group(h5_file["uns"]) if "uns" in h5_file and isinstance(h5_file["uns"], h5py.Group) else {}

    return ad.AnnData(X=mat, obs=obs, var=var, uns=uns)


def normalize_rna(
    adata,
    normalize_data=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=2000,
    pca=True,
    n_pcs=50,
    min_counts=3,
    min_genes=200,
    target_sum=1e4,
    max_value=10,
    seed=None,
):
    print(f"Original data shape: {adata.shape}")
    sc.pp.filter_genes(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"Filtered data shape: {adata.shape}")

    if normalize_data:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        if adata.X.max() > 20:
            sc.pp.log1p(adata)
            print("Applied log1p transformation.")
        else:
            print("Data appears to be log-transformed already. Skipping log1p.")

    if use_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat", batch_key=None)
        print(f"Selected highly variable genes: {adata.var['highly_variable'].sum()}")
        adata = adata[:, adata.var["highly_variable"]]
        if sp.issparse(adata.X):
            adata.obsm["hvg_data"] = adata.X.toarray()
        else:
            adata.obsm["hvg_data"] = adata.X.copy()
        if sp.issparse(adata.X):
            adata.uns["X_raw"] = adata.X.copy()

    if pca:
        adata_pca = adata.copy()
        sc.pp.scale(adata_pca, max_value=max_value, zero_center=True)
        sc.tl.pca(adata_pca, n_comps=n_pcs, svd_solver="auto", random_state=seed)
        adata.obsm["X_pca"] = adata_pca.obsm["X_pca"]
        print(f"PCA completed. Shape: {adata.obsm['X_pca'].shape}")

    return adata


def load_RNA_data(file_path):
    if file_path.endswith(".h5"):
        print(f"Detected .h5 file. Using custom loader: {file_path}")
        return read_h5_to_anndata(file_path)
    print(f"Loading standard .h5ad file: {file_path}")
    return sc.read_h5ad(file_path)


class RNADataset(Dataset):
    def __init__(self, file_path, n_hvgs=2000, cfg=None):
        if cfg is None or "seed" not in cfg:
            raise ValueError("RNADataset requires cfg with a seed value for reproducible results.")

        self.seed = cfg["seed"]
        print(f"RNADataset seed: {self.seed}")

        random.seed(self.seed)
        np.random.seed(self.seed)

        rna_data = load_RNA_data(file_path)

        data_cfg = cfg.get("data_config", {}) if cfg else {}
        graph_cfg = cfg.get("graph_config", {}) if cfg else {}
        module_cfg = cfg.get("module_config", {}) if cfg else {}

        print("=== Data diagnostics ===")
        print(f"Original data: {rna_data.shape}")
        if sp.issparse(rna_data.X):
            print("Sparse matrix detected.")
            data_stats = rna_data.X.data
            self.X_raw = rna_data.X.copy()
        else:
            print("Dense matrix detected.")
            data_stats = rna_data.X.flatten()
            self.X_raw = rna_data.X.copy()

        print(f"Data range: [{np.min(data_stats):.2f}, {np.max(data_stats):.2f}]")

        print("Preprocessing RNA data.")
        self.rna_data = normalize_rna(
            rna_data,
            normalize_data=True,
            log1p=True,
            use_hvg=True,
            n_top_genes=n_hvgs,
            pca=True,
            n_pcs=data_cfg.get("n_pcs", 50),
            min_counts=data_cfg.get("min_counts", 3),
            min_genes=data_cfg.get("min_genes", 200),
            target_sum=data_cfg.get("target_sum", 1e4),
            max_value=data_cfg.get("max_value", 10),
            seed=self.seed,
        )

        print("Data size (cells x genes): ", self.rna_data.obsm["hvg_data"].shape)
        self.n_cells = self.rna_data.obsm["hvg_data"].shape[0]
        self.n_genes = self.rna_data.obsm["hvg_data"].shape[1]

        print("Building SNN graph.")
        k_base = graph_cfg.get("k_base", 30)
        min_shared = graph_cfg.get("min_shared", 5)
        print(f"SNN config: k_base={k_base}, min_shared={min_shared}")

        X_pca = self.rna_data.obsm["X_pca"]
        nbrs = NearestNeighbors(n_neighbors=k_base, metric="euclidean").fit(X_pca)
        knn_graph = nbrs.kneighbors_graph(X_pca, mode="connectivity")

        snn_overlap = knn_graph * knn_graph.T
        snn_overlap.data[snn_overlap.data < min_shared] = 0
        snn_overlap.eliminate_zeros()
        snn_overlap.setdiag(0)

        self.W = normalize(snn_overlap, norm="l1", axis=1).toarray()

        avg_degree = np.sum(self.W > 0) / self.W.shape[0]
        print(f"SNN graph completed. Average degree: {avg_degree:.2f}")

        k_genes = module_cfg.get("k_genes", 10)
        leiden_res = module_cfg.get("res_gene_modules", 1.0)
        k_pb = module_cfg.get("k_pb_cells", 20)
        res_pb = module_cfg.get("res_pb_cells", 1.0)

        self.modules = self.find_gene_modules(
            k_neighbors=k_genes,
            leiden_resolution=leiden_res,
            pb_k_neighbors=k_pb,
            pb_resolution=res_pb,
        )

        raw_data = self.rna_data.obsm["hvg_data"].astype(np.float32)
        print("Standardizing data with StandardScaler.")
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(raw_data)
        print(f"Standardized data range: mean={np.mean(self.X):.4f}, std={np.std(self.X):.4f}")

        self.X_pca = self.rna_data.obsm["X_pca"].astype(np.float32)

        obs_keys = self.rna_data.obs.keys()
        target_key = None

        possible_keys = ["cell_type1", "cell_type", "Group", "subclass", "cluster"]
        for key in possible_keys:
            if key in obs_keys:
                target_key = key
                break

        if target_key is None and len(self.rna_data.obs.columns) > 0:
            if isinstance(self.rna_data.obs.columns[0], int):
                print("Warning: standard label column was not detected and columns are numeric.")
                print("Using the last column as labels.")
                target_key = self.rna_data.obs.columns[-1]

        if target_key is not None:
            print(f"Using label key: {target_key}")
            labels_raw = self.rna_data.obs[target_key].astype(str)
            self.labels = LabelEncoder().fit_transform(labels_raw)
            self.cell_types = self.rna_data.obs[target_key]
        else:
            print("Warning: labels were not found in obs. Accuracy metrics cannot be computed.")
            self.labels = None
            self.cell_types = None

    def find_gene_modules(
        self,
        k_neighbors=10,
        leiden_resolution=1.0,
        pb_k_neighbors=20,
        pb_resolution=1.0,
        min_cluster_size=5,
    ):
        hvgs = self.rna_data.var[self.rna_data.var["highly_variable"]].index

        sc.pp.neighbors(
            self.rna_data,
            n_neighbors=pb_k_neighbors,
            use_rep="X_pca",
            random_state=self.seed,
            transformer="sklearn",
        )
        sc.tl.leiden(self.rna_data, resolution=pb_resolution, random_state=self.seed)

        X = self.rna_data.obsm["hvg_data"]
        clusters = self.rna_data.obs["leiden"]
        total_cells = self.rna_data.shape[0]

        dynamic_threshold = max(min_cluster_size, int(total_cells * 0.005))

        cluster_counts = clusters.value_counts()
        valid_clusters = cluster_counts[cluster_counts >= dynamic_threshold].index.tolist()

        print("--- Adaptive module discovery ---")
        print(f"Total cells: {total_cells}")
        print(f"Dynamic threshold: >= {dynamic_threshold} cells")
        print(f"Original clusters: {len(cluster_counts)}")
        print(f"Valid clusters used for correlation: {len(valid_clusters)}")

        if len(valid_clusters) < 2:
            print("Warning: adaptive threshold was too strict. Falling back to the top 5 largest clusters.")
            valid_clusters = cluster_counts.index[:5].tolist()

        pb = pd.DataFrame(index=valid_clusters, columns=hvgs)
        for cl in valid_clusters:
            cell_mask = (clusters == cl).values
            if sp.issparse(X):
                pb.loc[cl] = X[cell_mask].mean(axis=0).A1
            else:
                pb.loc[cl] = X[cell_mask].mean(axis=0)

        expr = pb.T.values.astype(float)
        ranked = np.apply_along_axis(ss.rankdata, 1, expr)

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(ranked)
        corr = np.nan_to_num(corr)

        n_genes = corr.shape[0]
        adj_edges = []
        edge_weights = []
        for i in range(n_genes):
            idx = np.argsort(-corr[i, :])[: k_neighbors + 1]
            for j in idx:
                if j != i and corr[i, j] > 0:
                    adj_edges.append((i, j))
                    edge_weights.append(float(corr[i, j]))

        graph = ig.Graph(n=n_genes, edges=adj_edges)
        graph.es["weight"] = edge_weights

        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=graph.es["weight"],
            resolution_parameter=leiden_resolution,
            seed=self.seed,
        )
        labels = partition.membership
        print(f"Identified {len(set(labels))} gene modules.")

        gene_modules = pd.DataFrame({"gene": hvgs, "module": labels})
        gene2idx = {gene: i for i, gene in enumerate(hvgs)}
        modules = {}
        for module_id in sorted(set(labels)):
            modules[module_id] = [gene2idx[gene] for gene in hvgs[gene_modules.module == module_id]]

        return modules

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        return self.X[idx]
