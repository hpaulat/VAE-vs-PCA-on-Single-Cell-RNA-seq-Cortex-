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

Most of my current experience with data analysis has been with Seurat, so while the pipeline process was familiar to me, the actual implementation was a bit more difficult. I also don't have much experience coding machine learning models yet, so this required a lot of independent study.

I started by familiarizing myself with [ScanPy](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html), and the [AnnData](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) object structure. I followed the ScanPy tutorial to normalize and log the counts data. To focus on informative features, I selected 2,000 HVGs and subsetted the matrix so the model only analyzes those genes.

I implemented an encoder MLP (2000 → 256 → 64), two linear heads for μ and logσ², and a mirrored decoder (latent → 64 → 256 → 2000). The choice for the number of neurons per layer was purely experimental, I tried a multitude of values to see what worked best. I leaned on PyTorch’s [nn.Module](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) docs for model structure and the [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html?utm_source=chatgpt.com) tutorial for feeding batches cleanly. This [repo](https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb) by Jackson Kang also helped immensely.

The training loop uses Adam, mini-batches from DataLoader, and a standard VAE loss: MSE reconstruction (on normalized/log1p data) plus KL divergence between 𝑞𝜙(𝑧∣𝑥)and 𝑁(0,𝐼). Switching the model to train() during optimization and eval() for inference follows PyTorch best practices.

After training, I switched to model.eval() and iterated through the dataset with torch.no_grad() and shuffle=False to preserve row order, collecting posterior means μ into a dense matrix Z of shape (n_cells, d_latent). The DataLoader tutorial was helpful for the “dataset → iterable of batches” pattern.

I then ran: _umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0).fit_transform(Z)_
to embed to 2-D and then colored points by ground-truth labels from adata.obs["label"]. The [UMAP docs](https://umap-learn.readthedocs.io/en/latest/parameters.html?utm_source=chatgpt.com) guided parameter choices (e.g., how min_dist affects cluster tightness and what random_state controls).

Before clustering I standardized Z per dimension (StandardScaler) because K-means is distance-based. Then I ran KMeans. I computed Adjusted Rand Index (ARI) with _sklearn.metrics.adjusted_rand_score(y_true, y_pred)_ to compare K-means labels against the known cell-type labels; the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?utm_source=chatgpt.com) docs explain both K-means parameters (like n_init) and ARI.

For a like-for-like baseline, I computed PCA on the same HVG-filtered matrix (matching the VAE latent size), then repeated the same K-means + ARI procedure on the PCA scores.

## Lessons Learned

Something that really helped me with this project was just throwing myself into it and learning as I went. As I mentioned, I don't have a ton of experience with coding neural networks, and the knowledge I do have is much more general than VAE's specifically. This made this project a bit daunting at first. Still, once I threw myself into it I felt like everything became much more manageable. The best way to learn is by doing. I was really happy to code in Python as well, since it had been a little while.

## Helpful Resources

https://medium.com/@weidagang/demystifying-neural-networks-variational-autoencoders-6a44e75d0271

https://www.youtube.com/watch?v=9zKuYvjFFS8

https://www.youtube.com/watch?v=uvyG9yLuNSE

https://academic.oup.com/bioinformatics/article/36/16/4415/5838187
