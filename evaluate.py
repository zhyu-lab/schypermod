"""Evaluate scHyperMod embeddings with Leiden clustering."""

from __future__ import annotations

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
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder


CONFIG = {
    "seed": 3333,
    "eval_config": {
        "large_data_threshold": 3000,
        "large_k": 30,
        "small_k": 10,
        "resolution_search_range": np.arange(0.001, 2.5, 0.005),
    },
}


def cluster_acc(y_true, y_pred):
    """Compute clustering accuracy after optimal Hungarian label matching."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same number of elements.")
    if y_true.size == 0:
        raise ValueError("Cannot compute clustering accuracy for empty labels.")

    dim = max(y_pred.max(), y_true.max()) + 1
    weight = np.zeros((dim, dim), dtype=np.int64)

    for pred, true in zip(y_pred, y_true):
        weight[pred, true] += 1

    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    return weight[row_ind, col_ind].sum() / y_pred.size


def compute_clustering_metrics(y_true, y_pred):
    """Compute ARI, NMI, AMI, and clustering accuracy."""
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    acc = cluster_acc(y_true, y_pred)

    return [ari, nmi, ami, acc]


def resolve_adata_path(adata_path=None, h5ad_dir="h5ad", dataset_name=None):
    """Resolve the input AnnData path."""
    if adata_path is None:
        if dataset_name is None:
            raise ValueError("Either adata_path or dataset_name must be provided.")
        adata_path = os.path.join(h5ad_dir, f"{dataset_name}.h5ad")

    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    return adata_path


def infer_dataset_name_from_path(path):
    """Infer a dataset name from an AnnData file path."""
    return os.path.splitext(os.path.basename(path))[0].replace("_evaluated", "")


def get_dataset_name(adata, adata_path, dataset_name=None):
    """Get the dataset name from arguments, metadata, or file name."""
    if dataset_name is not None:
        return dataset_name

    if "dataset_name" in adata.uns:
        return str(adata.uns["dataset_name"])

    return infer_dataset_name_from_path(adata_path)


def select_embedding_key(adata, requested_key=None):
    """Select the embedding key used for clustering evaluation."""
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

    raise KeyError(
        "No embedding key found. Pass --embedding-key or store embeddings in "
        "adata.obsm."
    )


def get_true_labels(adata, label_key=None, cell_type_key="cell_type"):
    """Return integer-encoded ground-truth labels and the label key used."""
    if label_key is not None:
        if label_key not in adata.obs:
            raise KeyError(f"Label key not found in adata.obs: {label_key}")

        values = np.asarray(adata.obs[label_key])

        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.int64), label_key

        return LabelEncoder().fit_transform(values.astype(str)), label_key

    if "label_encoded" in adata.obs:
        return (
            np.asarray(adata.obs["label_encoded"]).astype(np.int64),
            "label_encoded",
        )

    if cell_type_key in adata.obs:
        values = np.asarray(adata.obs[cell_type_key].astype(str))
        return LabelEncoder().fit_transform(values), cell_type_key

    return None, None


def h5ad_metrics_table(result_rows):
    """Convert evaluation rows into a mapping suitable for AnnData metadata."""
    if not result_rows:
        return {}

    frame = pd.DataFrame(result_rows)
    output = {}

    for column in frame.columns:
        series = frame[column]

        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            output[column] = series.to_numpy()
        else:
            output[column] = series.fillna("").astype(str).to_numpy()

    return output


def evaluate_anndata(
    adata,
    cfg=None,
    dataset_name=None,
    embedding_key=None,
    label_key=None,
    cell_type_key="cell_type",
    output_dir="outputs",
):
    """
    Evaluate embeddings with Leiden clustering.

    Resolutions are searched from large to small with a coarse-to-fine strategy.
    Coarse search uses a step of 0.025. Once a coarse result reaches K + 2
    clusters or fewer, the search switches to the original 0.005-resolution
    grid, with a small upward buffer, and continues from large to small.

    Selection priority:
    1. Select the highest tested resolution producing exactly K clusters,
       where K is the number of ground-truth classes.
    2. If no exact K-cluster result exists, search the observed cluster counts
       in the order K+1, K-1, K+2, K-2, ... until an available count is found.
       For that cluster count, select the largest corresponding resolution.

    ARI, NMI, AMI, and ACC are reported for the selected result.
    """
    if cfg is None:
        cfg = CONFIG

    os.makedirs(output_dir, exist_ok=True)

    embedding_key = select_embedding_key(adata, embedding_key)
    embeddings = np.asarray(adata.obsm[embedding_key], dtype=np.float32)

    eval_cfg = cfg["eval_config"]
    num_cells = embeddings.shape[0]

    if embeddings.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got shape {embeddings.shape}.")

    if num_cells != adata.n_obs:
        raise ValueError(
            f"Embedding row count {num_cells} does not match AnnData n_obs "
            f"{adata.n_obs}."
        )

    if num_cells < 2:
        raise ValueError("At least two cells are required for evaluation.")

    if num_cells > eval_cfg["large_data_threshold"]:
        eval_neighbors = eval_cfg["large_k"]
    else:
        eval_neighbors = eval_cfg["small_k"]

    eval_neighbors = min(int(eval_neighbors), num_cells - 1)

    true_labels, used_label_key = get_true_labels(
        adata,
        label_key=label_key,
        cell_type_key=cell_type_key,
    )

    if true_labels is None:
        raise ValueError(
            "Ground-truth labels are required. Provide --label-key or ensure "
            "label_encoded/cell_type exists in adata.obs."
        )

    true_n_clusters = len(np.unique(true_labels))

    if true_n_clusters < 1:
        raise ValueError("No valid ground-truth classes were found.")

    if cell_type_key not in adata.obs:
        adata.obs[cell_type_key] = np.asarray(true_labels).astype(str)
        adata.obs[cell_type_key] = adata.obs[cell_type_key].astype("category")

    neighbors_key = "masked_clustering_neighbors"
    selected_cluster_key = "leiden_selected"
    temp_cluster_key = "leiden_tmp"

    print(f"Loaded AnnData with shape: {adata.shape}")
    print(f"Using embedding key: {embedding_key}")
    print(f"Using ground-truth label key: {used_label_key}")
    print(f"True number of classes: {true_n_clusters}")
    print(f"Evaluation n_neighbors={eval_neighbors} for {num_cells} cells.")

    print("Computing neighbors and UMAP.")

    sc.pp.neighbors(
        adata,
        use_rep=embedding_key,
        n_neighbors=eval_neighbors,
        metric="cosine",
        key_added=neighbors_key,
    )

    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        random_state=cfg["seed"],
    )

    resolutions = np.asarray(
        eval_cfg["resolution_search_range"],
        dtype=float,
    )

    if resolutions.size == 0:
        raise ValueError("resolution_search_range is empty.")

    resolutions = np.sort(resolutions)[::-1]

    if resolutions.size > 1:
        fine_step = float(np.min(np.abs(np.diff(resolutions))))
    else:
        fine_step = 0.005

    coarse_step = 0.025
    coarse_stride = max(1, int(round(coarse_step / fine_step)))

    selected_metrics = None
    selected_res = None
    selected_n_clusters = None
    selected_labels = None
    selection_rule = None

    best_by_cluster_count = {}
    evaluated_by_resolution = {}

    result_rows = []

    print("Searching Leiden resolution from large to small.")
    print(
        f"Coarse step={coarse_step:.3f}; fine step={fine_step:.3f}. "
        "Fine search starts when a coarse result reaches K+2 clusters or fewer."
    )
    print(
        "The search stops at the highest fine-grid resolution producing "
        "the true class count."
    )
    print(
        "If no exact match exists, observed cluster counts are checked in the "
        "order K+1, K-1, K+2, K-2, ... until a match is available."
    )
    print(
        "For the selected fallback cluster count, the largest corresponding "
        "evaluated resolution is used."
    )

    print("-" * 70)
    print(
        f"{'Resolution':<12} | "
        f"{'Clusters':<10} | "
        f"{'Target':<10} | "
        f"{'Exact':<7}"
    )
    print("-" * 70)

    def evaluate_resolution(res):
        res = float(res)
        res_key = round(res, 12)

        if res_key in evaluated_by_resolution:
            return evaluated_by_resolution[res_key]

        try:
            sc.tl.leiden(
                adata,
                key_added=temp_cluster_key,
                resolution=res,
                random_state=cfg["seed"],
                neighbors_key=neighbors_key,
            )

            labels_p = (
                adata.obs[temp_cluster_key]
                .astype(int)
                .to_numpy()
            )

            labels_series = adata.obs[temp_cluster_key].copy()

            n_clusters = len(np.unique(labels_p))
            is_match = n_clusters == true_n_clusters

            row = {
                "resolution": res,
                "n_clusters": int(n_clusters),
                "true_n_clusters": int(true_n_clusters),
                "is_match": bool(is_match),
                "ari": np.nan,
                "nmi": np.nan,
                "ami": np.nan,
                "acc": np.nan,
                "error": "",
            }

            row_index = len(result_rows)
            result_rows.append(row)

            record = {
                "resolution": res,
                "n_clusters": int(n_clusters),
                "labels_array": labels_p.copy(),
                "labels_series": labels_series.copy(),
                "row_index": row_index,
            }

            evaluated_by_resolution[res_key] = record

            if n_clusters not in best_by_cluster_count:
                best_by_cluster_count[n_clusters] = record
            elif res > best_by_cluster_count[n_clusters]["resolution"]:
                best_by_cluster_count[n_clusters] = record

            print(
                f"{res:<12.5f} | "
                f"{n_clusters:<10d} | "
                f"{true_n_clusters:<10d} | "
                f"{str(is_match):<7}"
            )

            return record

        except Exception as err:
            print(f"Skipping resolution {res:.5f}: {err}")

            result_rows.append(
                {
                    "resolution": res,
                    "n_clusters": np.nan,
                    "true_n_clusters": int(true_n_clusters),
                    "is_match": False,
                    "ari": np.nan,
                    "nmi": np.nan,
                    "ami": np.nan,
                    "acc": np.nan,
                    "error": str(err),
                }
            )

            evaluated_by_resolution[res_key] = None
            return None

    def select_exact_record(record):
        metrics = compute_clustering_metrics(
            true_labels,
            record["labels_array"],
        )

        ari, nmi, ami, acc = metrics

        result_rows[record["row_index"]].update(
            {
                "ari": float(ari),
                "nmi": float(nmi),
                "ami": float(ami),
                "acc": float(acc),
            }
        )

        return metrics

    coarse_indices = list(range(0, len(resolutions), coarse_stride))
    if coarse_indices[-1] != len(resolutions) - 1:
        coarse_indices.append(len(resolutions) - 1)

    fine_trigger_index = None

    for resolution_index in coarse_indices:
        record = evaluate_resolution(resolutions[resolution_index])

        if record is None:
            continue

        if record["n_clusters"] <= true_n_clusters + 2:
            fine_trigger_index = resolution_index
            break

    if fine_trigger_index is None:
        print(
            "K+2 or fewer clusters were not reached during coarse search. "
            "Falling back to the complete fine-resolution grid."
        )
        fine_start_index = 0
    else:
        fine_start_index = max(
            0,
            fine_trigger_index - 2 * coarse_stride,
        )

        print(
            "Switching to fine search from "
            f"resolution={resolutions[fine_start_index]:.5f}."
        )

    for resolution_index in range(fine_start_index, len(resolutions)):
        record = evaluate_resolution(resolutions[resolution_index])

        if record is None:
            continue

        if record["n_clusters"] == true_n_clusters:
            selected_metrics = select_exact_record(record)
            selected_res = record["resolution"]
            selected_n_clusters = record["n_clusters"]
            selected_labels = record["labels_series"]
            selection_rule = "exact_true_cluster_count"

            print("-" * 70)
            print(
                f"Exact match found at resolution={selected_res:.5f}; "
                f"predicted clusters={selected_n_clusters}, "
                f"true classes={true_n_clusters}."
            )
            print("Stopping resolution search.")

            break

    if temp_cluster_key in adata.obs:
        del adata.obs[temp_cluster_key]

    suffix = f"_{dataset_name}" if dataset_name else ""

    umap_path = os.path.join(
        output_dir,
        f"umap_result{suffix}.png",
    )

    metrics_path = os.path.join(
        output_dir,
        f"metrics{suffix}.csv",
    )

    if selected_labels is None:
        fallback = None
        fallback_target = None

        max_delta = max(
            true_n_clusters - 1,
            max(best_by_cluster_count.keys(), default=true_n_clusters)
            - true_n_clusters,
        )

        for delta in range(1, max_delta + 1):
            higher_count = true_n_clusters + delta
            lower_count = true_n_clusters - delta

            if higher_count in best_by_cluster_count:
                fallback = best_by_cluster_count[higher_count]
                fallback_target = higher_count
                break

            if lower_count >= 1 and lower_count in best_by_cluster_count:
                fallback = best_by_cluster_count[lower_count]
                fallback_target = lower_count
                break

        if fallback is not None:
            selection_rule = "closest_cluster_count_prefer_higher_then_highest_resolution"

            print("-" * 70)
            print("No exact cluster-count match was found.")
            print(
                f"Selecting fallback cluster count {fallback_target}, following "
                "the order K+1, K-1, K+2, K-2, ..."
            )
            print(
                "Using the largest resolution observed for that cluster count."
            )

            selected_res = fallback["resolution"]
            selected_n_clusters = fallback["n_clusters"]
            selected_labels = fallback["labels_series"]

            selected_metrics = compute_clustering_metrics(
                true_labels,
                fallback["labels_array"],
            )

            ari, nmi, ami, acc = selected_metrics

            result_rows[fallback["row_index"]].update(
                {
                    "ari": float(ari),
                    "nmi": float(nmi),
                    "ami": float(ami),
                    "acc": float(acc),
                }
            )

            print(
                f"Selected fallback resolution={selected_res:.5f}; "
                f"predicted clusters={selected_n_clusters}, "
                f"true classes={true_n_clusters}."
            )

    metrics_df = pd.DataFrame(result_rows)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved resolution search table to: {metrics_path}")

    if selected_labels is None:
        raise RuntimeError(
            "No valid Leiden clustering result was produced in the configured "
            "resolution search range. "
            f"Search results were saved to: {metrics_path}"
        )

    adata.obs[selected_cluster_key] = selected_labels.astype("category")

    ari, nmi, ami, acc = selected_metrics

    print("=" * 72)
    print("Selected evaluation result")
    print(f"Selection rule: {selection_rule}")
    print(f"Resolution: {selected_res:.5f}")
    print(f"True classes: {true_n_clusters}")
    print(f"Predicted clusters: {selected_n_clusters}")
    print(f"ARI: {ari:.5f}")
    print(f"NMI: {nmi:.5f}")
    print(f"AMI: {ami:.5f}")
    print(f"ACC: {acc:.5f}")
    print("=" * 72)

    print("Generating UMAP plot.")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    sc.pl.umap(
        adata,
        color=[cell_type_key],
        title=f"Ground Truth (Classes={true_n_clusters})",
        show=False,
        ax=ax1,
    )

    sc.pl.umap(
        adata,
        color=[selected_cluster_key],
        title=(
            f"Predicted (Res={selected_res:.5f}, "
            f"Clusters={selected_n_clusters}, "
            f"ARI={ari:.5f})"
        ),
        show=False,
        ax=ax2,
    )

    plt.tight_layout()

    plt.savefig(
        umap_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    selected_metrics_dict = {
        "ari": float(ari),
        "nmi": float(nmi),
        "ami": float(ami),
        "acc": float(acc),
    }

    adata.uns["masked_clustering_eval"] = {
        "stage": "evaluated",
        "dataset_name": (
            str(dataset_name)
            if dataset_name
            else "unknown"
        ),
        "embedding_key": str(embedding_key),
        "neighbors_key": neighbors_key,
        "cluster_key": selected_cluster_key,
        "label_key": str(used_label_key),
        "true_n_clusters": int(true_n_clusters),
        "selected_resolution": float(selected_res),
        "selected_n_clusters": int(selected_n_clusters),
        "selection_rule": selection_rule,
        "metrics": selected_metrics_dict,
        "metrics_table": h5ad_metrics_table(result_rows),
        "umap_path": umap_path,
        "metrics_path": metrics_path,
        "best_resolution": float(selected_res),
        "best_n_clusters": int(selected_n_clusters),
        "best_metrics": selected_metrics_dict,
        "selected_by": selection_rule,
    }

    print(f"Saved UMAP plot to: {umap_path}")

    return (
        adata,
        selected_metrics,
        selected_n_clusters,
        umap_path,
        metrics_path,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved AnnData object with Leiden clustering. "
            "Resolutions are searched from large to small using a coarse-to-fine "
            "strategy with coarse step 0.025 and the original fine step 0.005. "
            "Fine search starts when a coarse result reaches K+2 clusters or fewer. "
            "The highest fine-grid resolution yielding the target cluster count is "
            "selected. If no exact match exists, observed cluster counts are checked "
            "in the order K+1, K-1, K+2, K-2, ... until an available count is found, "
            "and the largest corresponding evaluated resolution is selected."
        )
    )

    parser.add_argument(
        "--adata-path",
        type=str,
        default=None,
        help="Direct path to a saved .h5ad file.",
    )

    parser.add_argument(
        "--h5ad-dir",
        type=str,
        default="embeddings",
        help="Directory containing saved .h5ad files.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Dataset name used to locate "
            "<h5ad-dir>/<dataset_name>.h5ad."
        ),
    )

    parser.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help=(
            "Embedding key in adata.obsm. "
            "Defaults to the key saved by train.py."
        ),
    )

    parser.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="Ground-truth label key in adata.obs.",
    )

    parser.add_argument(
        "--cell-type-key",
        type=str,
        default="cell_type",
        help="Cell type key used for the ground-truth UMAP panel.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for plots and metric tables.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path for the evaluated .h5ad output file.",
    )

    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help="Overwrite the input .h5ad with evaluated results.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3333,
        help="Random seed.",
    )

    parser.add_argument(
        "--large-data-threshold",
        type=int,
        default=3000,
        help="Cell-count threshold for using large_k.",
    )

    parser.add_argument(
        "--large-k",
        type=int,
        default=30,
        help="n_neighbors for large datasets.",
    )

    parser.add_argument(
        "--small-k",
        type=int,
        default=10,
        help="n_neighbors for small datasets.",
    )

    parser.add_argument(
        "--resolution-start",
        type=float,
        default=0.001,
        help="Resolution search start.",
    )

    parser.add_argument(
        "--resolution-stop",
        type=float,
        default=2.5,
        help="Resolution search stop (exclusive).",
    )

    parser.add_argument(
        "--resolution-step",
        type=float,
        default=0.005,
        help="Resolution search step.",
    )

    return parser.parse_args()


def build_runtime_config(args):
    """Build runtime evaluation configuration from CLI arguments."""
    if args.resolution_step <= 0:
        raise ValueError(
            "--resolution-step must be greater than 0."
        )

    if args.resolution_stop <= args.resolution_start:
        raise ValueError(
            "--resolution-stop must be greater than "
            "--resolution-start."
        )

    if args.large_k < 1 or args.small_k < 1:
        raise ValueError(
            "--large-k and --small-k must be at least 1."
        )

    return {
        "seed": args.seed,
        "eval_config": {
            "large_data_threshold": args.large_data_threshold,
            "large_k": args.large_k,
            "small_k": args.small_k,
            "resolution_search_range": np.arange(
                args.resolution_start,
                args.resolution_stop,
                args.resolution_step,
            ),
        },
    }


def resolve_save_path(args, input_path, dataset_name):
    """Resolve the output path for the evaluated AnnData file."""
    if args.overwrite_input:
        return input_path

    if args.save_path is not None:
        return args.save_path

    base_dir = os.path.dirname(input_path) or "."

    return os.path.join(
        base_dir,
        f"{dataset_name}_evaluated.h5ad",
    )


def main():
    """Run evaluation from the command line."""
    args = parse_args()

    runtime_cfg = build_runtime_config(args)

    input_path = resolve_adata_path(
        adata_path=args.adata_path,
        h5ad_dir=args.h5ad_dir,
        dataset_name=args.dataset_name,
    )

    loaded_adata = ad.read_h5ad(input_path)

    loaded_dataset_name = get_dataset_name(
        loaded_adata,
        input_path,
        args.dataset_name,
    )

    (
        evaluated_adata,
        metrics,
        n_clusters,
        _,
        _,
    ) = evaluate_anndata(
        loaded_adata,
        cfg=runtime_cfg,
        dataset_name=loaded_dataset_name,
        embedding_key=args.embedding_key,
        label_key=args.label_key,
        cell_type_key=args.cell_type_key,
        output_dir=args.output_dir,
    )

    output_h5ad_path = resolve_save_path(
        args,
        input_path,
        loaded_dataset_name,
    )

    output_parent = os.path.dirname(output_h5ad_path)

    if output_parent:
        os.makedirs(
            output_parent,
            exist_ok=True,
        )

    evaluated_adata.write_h5ad(
        output_h5ad_path,
        compression="gzip",
    )

    print(
        f"Saved evaluated AnnData to: "
        f"{output_h5ad_path}"
    )

    print("Final Results:")
    print(f"ARI: {metrics[0]:.5f}")
    print(f"NMI: {metrics[1]:.5f}")
    print(f"AMI: {metrics[2]:.5f}")
    print(f"ACC: {metrics[3]:.5f}")
    print(
        f"Number of predicted clusters: "
        f"{n_clusters}"
    )
    print(
        f"Evaluated AnnData file: "
        f"{output_h5ad_path}"
    )


if __name__ == "__main__":
    main()