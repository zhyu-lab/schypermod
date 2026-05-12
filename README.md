# scHyperMod

Hypergraph-Enhanced Self-Supervised Learning with Module-Aware Structured Masking for Single-Cell Transcriptomic Clustering.

## Requirements

* Python 3.10+.
* CUDA-enabled GPU is recommended for training.

# Installation

## Clone repository

First, download scHyperMod from github and change to the directory:

```bash
git clone https://github.com/zhyu-lab/schypermod
cd schypermod
```

## Create conda environment (optional)

Create a new environment named "schypermod":

```bash
conda create --name schypermod python=3.10
```

Then activate it:

```bash
conda activate schypermod
```

## Install requirements

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

# Usage

## Step 1: Prepare the input data in h5ad or h5 format.

We use single-cell RNA-seq data stored in .h5ad or .h5 files as input.

The default dataset organization is:

```bash
data/<dataset_name>/data.h5ad
```

Alternatively, a direct input file path can be provided with `--rna-path`.

## Step 2: Run scHyperMod

The `train.py` Python script is used to train the model and save the learned cell embeddings as an AnnData object.

Example:

```bash
python train.py --rna-path ./data/Adam/data.h5 --dataset-name Adam --gpu 0 --h5ad-dir ./embeddings
```

The main arguments are as follows:

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| --rna-path | input file containing RNA data | ./data/Adam/data.h5 |
| --dataset-name | dataset name used for output naming | Adam |
| --gpu | GPU index | 0 |
| --h5ad-dir | directory to save trained AnnData files | ./embeddings |

This command saves the trained AnnData file to:

```bash
./embeddings/Adam.h5ad
```

Other model hyperparameters are fixed in the `CONFIG` dictionary in `train.py`.

## Step 3: Evaluate scHyperMod

The `evaluate.py` Python script is used to evaluate the learned embeddings with Leiden clustering and save the evaluation results.

Example:

```bash
python evaluate.py --dataset-name Adam --h5ad-dir ./embeddings --output-dir ./outputs
```

The main arguments are as follows:

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| --dataset-name | dataset name used to locate the saved .h5ad file | Adam |
| --h5ad-dir | directory containing saved .h5ad files | ./embeddings |
| --output-dir | directory for UMAP plots and metric tables | ./outputs |

This command reads the trained AnnData file from:

```bash
./embeddings/Adam.h5ad
```

The evaluation script saves the UMAP plot and metrics table to:

```bash
./outputs/umap_result_Adam.png
./outputs/metrics_Adam.csv
```

The evaluated AnnData file is saved to:

```bash
./embeddings/Adam_evaluated.h5ad
```

If ground-truth labels are available in `adata.obs`, scHyperMod reports ARI, NMI, AMI, ACC, and SIL. Otherwise, it reports SIL.

Other evaluation hyperparameters are fixed in the `CONFIG` dictionary in `evaluate.py`.

# Contact

If you have any questions, please contact cyw_nxu@163.com.
