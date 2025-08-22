# VAE vs PCA on Single-Cell RNA-seq (Cortex)

This repo trains a small Variational Autoencoder (VAE) on a single-cell RNA-seq dataset (`.h5ad`), compares it to PCA, and evaluates clustering with ARI. It also produces UMAP visualizations colored by ground-truth cell types. I wrote this as an entry test for a lab at McGill University.

## How to Run

### Conda

```bash
conda create -n scanpyLabTest python=3.10 -y
conda activate scanpyLabTest
pip install -r requirements.txt
```

## How did I build it?

Most of my previous single-cell work was in Seurat, so I first focused on learning Scanpy/AnnData basics and mapped the pipeline I was used to into Python:

### Preprocessing

```
# Save raw counts to adata.layers["counts"]

sc.pp.normalize_total
sc.pp.log1p
sc.pp.highly_variable_genes(n_top_genes=2000, subset=True)
```

### Per-Gene Weighting for Reconstruction.

I compute weights from normalized dispersion and rescale them to keep the average weight ≈ 1, so the weighted MSE remains comparable to unweighted MSE. A single knob gamma controls how strongly HVGs are emphasized.

### Model

- Encoder MLP: input → 1024 → 256 → 64 with GELU and LayerNorm
- Two heads: μ and log(σ²)
- Decoder (mirrored): latent → 64 → 256 → 1024 → input
- Latent size: d_latent = 6

### Objective & schedule.

- Weighted MSE (on log-normalized HVGs) + KL
- β-VAE cosine warm-up to beta_max (default 0.1) over warmup epochs
- Adam (lr=1e-3), batch size 256, default 50 epochs

### Evaluation.

- Collect posterior means μ for all cells, Standardize, then KMeans with K = #unique labels
- ARI between predicted clusters and true labels
- Baseline: run PCA (same HVG matrix; keep first n_pcs) → Standardize → KMeans → ARI
- Visualization: UMAP(μ) and UMAP(PCA) side-by-side, colored by adata.obs["label"] with a shared colormap

### Reproducibility.

Seeds are set across numpy, Python random, and PyTorch; deterministic flags are enabled

### Choices

- HVG-weighted reconstruction loss.
  Emphasizes biologically variable genes while keeping overall loss scale stable (average weight ≈ 1).
- β-annealing via cosine schedule.
  Warms in the KL term to avoid early latent collapse; try beta_max in range [0.1, 0.5] and warmup in range [10, 100]
- Reporting.
  Epoch logs report objective/recon/β·KL per cell (matching what’s optimized) and ARI. Final print shows [Final] ARI VAE and ARI PCA(n_pcs).

### Knobs to tune quickly

- d_latent (e.g., 4/8/12)
- beta_max, warmup (β schedule)
- gamma (HVG weight strength)
- n_pcs for the PCA baseline
- n_neighbors, min_dist in UMAP

### Future work

- Validation split + early stopping on ARI(val). Track ARI(val) and keep the best checkpoint.
- Swap MSE for NB on raw counts to model over-dispersion; keep the current log-normalized MSE path as a baseline.
- Cluster–label heatmaps.
- Graph Training

## Lessons Learned

Jumping from Seurat to a PyTorch VAE in Scanpy was a bit difficult at first, especially due to me having less experience coding neural networks. It took some time, but thankfully there were a plethora of resources and documentation online to help me get started (see below). Working through everything really deepend my understanding in a lot of places, which I think will really help me as the new school year starts (I'm taking a lot of machine-learning intensive courses). There is still a lot of ways that I can improve on my work, but I'm really happy with the progress I was able to make in this amount of time.

## Helpful Resources

https://medium.com@weidagangdemystifying-neural-networks-variational-autoencoders-6a44e75d0271

https://www.youtube.com/watch?v=9zKuYvjFFS8

https://www.youtube.com/watch?v=uvyG9yLuNSE

https://academic.oup.com/bioinformatics/article/36/16/4415/5838187

https://github.com/lasersonlab/single-cell-experiments/blob/master/README.md
