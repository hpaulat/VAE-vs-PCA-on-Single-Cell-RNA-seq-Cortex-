import scanpy as sc
import anndata as ad
import math, numpy as np
import torch, torch.nn as nn, torch.utils.data as tud
import umap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os, random, numpy as np, torch

SEED = 0  # CHATGPT code to keep runs consistent
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

adata = sc.read_h5ad("/Users/hugopaulat/Desktop/projects/scanPyLabTest/cortex.h5ad")
print("General AnnObject Info")
print(adata)
print("Per-cell metadata")
print(adata.obs)
print ("Per-gene metadata")
print(adata.var)

# Quality Control
    # calculate_qc_metrics()
    # Doublet detection

# --------- NORMALIZATION ---------
adata.layers["counts"] = adata.X.copy()         # store counts in layers
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)


# --------- FEATURE SELECTION ---------
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)       # Subsetted already so no need to visualize
print("[HVG] using", adata.n_vars, "genes")

# Calculating Weights
disp = adata.var["dispersions_norm"].to_numpy()         # Per-gene normalized dispersion
disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)      # Finds low quality values and sets to 0 --> asserts float32 type

gamma = 3      # main knob
rng = np.ptp(disp) if np.ptp(disp) != 0 else 1.0  # calculates range for each gene
w = 1.0 + gamma * (disp - disp.min()) / (rng + 1e-8)  # weights >= 1 for more variable genes
w = w.astype(np.float32)

# Keep overall loss scale comparable to unweighted MSE (avg weight ~ 1)
w *= (w.size / w.sum())


# --------- VAE MODEL ---------
X = np.asarray(adata.X, dtype=np.float32) 
print(f"[Check] X dimensions = {X.shape}, dtype = {X.dtype}, type = {type(X)}")

# Data Loader
batch_size = 256
ds = tud.TensorDataset(torch.from_numpy(X))
g = torch.Generator(device="cpu").manual_seed(SEED)         # change to cuda for graphics cards
dl = tud.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
print(f"[Sanity] n_cells = {len(ds)}, n_genes = {X.shape[1]},"
      f"batches/epoch ≈ {math.ceil(len(ds)/batch_size)}")

# VAE 
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):      # Constructor (# input features, latent dimensionality)
        super().__init__()
        self.enc = nn.Sequential(       # Initializes two fully connected layers, ReLu Activation Function
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU()
        )
        self.mu = nn.Linear(64, latent_dim)        # Predict mean vector
        self.logvar = nn.Linear(64, latent_dim)        # Predict log-variance
        self.dec = nn.Sequential(       # Mirrors the encoder but back to the input space
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x): # Encoder head (input cell vector x --> the parameters of a Gaussian over the latent z)
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
    
    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, gene_w):
    diff2 = (x_hat - x) ** 2
    recon = (diff2 * gene_w).sum() if gene_w is not None else diff2.sum()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

# Training
d_in, d_latent = X.shape[1], 8         # input features and latent dimensionality for VAE
device = "cpu"                                     
model = VAE(d_in, d_latent).to(device)          # moves parameters to evice
opt = torch.optim.Adam(model.parameters(), lr=1e-3)         # solid default for Adam optimizer

# --- Make a torch tensor for the per-gene weights ---
gene_w = torch.from_numpy(w).to(device)   # shape (d_in,)

epochs = 50
print(f"[Info] Training on {device} for {epochs} epochs...")
for epoch in range(1, epochs + 1):
    model.train()           # good practice    
    total, total_recon, total_kl = 0.0, 0.0, 0.0        # Accumulators to track summed losses across the epoch
    for (xb,) in dl:    
        xb = xb.to(device)          # Loads one mini-batch to device 
        opt.zero_grad()         # Clears gradients from last iteration
        x_hat, mu, logvar = model(xb)
        loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, gene_w=gene_w)
        loss.backward()         # Backpropagation
        opt.step()          # Optimizer update
        total += loss.item()
        total_recon += recon.item()         # Track losses
        total_kl += kl.item()
    print(f"Epoch {epoch:03d} | loss/cell={total/len(ds):.2f} "
          f"(recon/cell={total_recon/len(ds):.2f}, kl/cell={total_kl/len(ds):.2f})")

# Extract latent Means VAE
model.eval()        # Evaluation mode
mu_chunks = []      # Collector list for batches of encoder means μ
with torch.no_grad():       # Disables gradient tracking
    for (xb,) in tud.DataLoader(ds, batch_size=1024, shuffle=False):        # Iterates over all cells
        xb = xb.to(device)
        mu, _ = model.encode(xb)        # Only runs encoder (not full forward pass)
        mu_chunks.append(mu.cpu().numpy())      # Stitches all values into matrix

Z = np.concatenate(mu_chunks, axis=0)       # Concatenates the per-batch arrays along rows (axis 0) to form the full latent matrix.
print("[Check] Z shape:", Z.shape)      # Sanity Check       

# Labels
label_col = "label"
labels = adata.obs[label_col].astype("category")
y_true = labels.cat.codes.to_numpy()
K = len(labels.cat.categories)

# --------- PCA ---------
n_pcs = 16  # match your d_latent=16; you can also try 32/50 and compare
sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=SEED)
X_pca = adata.obsm["X_pca"][:, :n_pcs]        # (n_cells, n_pcs)

# --------- UMAP ---------
def umap_embed(X, seed=0):
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED).fit_transform(X)

emb_vae = umap_embed(Z)         # UMAP of VAE latent
emb_pca = umap_embed(X_pca)     # UMAP of PCA space
codes = labels.cat.codes.to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

axes[0].scatter(emb_vae[:,0], emb_vae[:,1], s=6, c=codes, cmap="tab20", rasterized=True)
axes[0].set_title("VAE latent → UMAP"); axes[0].set_xlabel("UMAP1"); axes[0].set_ylabel("UMAP2")
axes[1].scatter(emb_pca[:,0], emb_pca[:,1], s=6, c=codes, cmap="tab20", rasterized=True)
axes[1].set_title(f"PCA ({n_pcs} PCs) → UMAP"); axes[1].set_xlabel("UMAP1"); axes[1].set_ylabel("UMAP2")

plt.tight_layout(); plt.show()

# --------- K-means + ARI Comparison ---------
X_pca_std = StandardScaler().fit_transform(X_pca)  # helps K-means
km_pca = KMeans(n_clusters=K, n_init=100, random_state=SEED)
y_pred_pca = km_pca.fit_predict(X_pca_std)
ari_pca = adjusted_rand_score(y_true, y_pred_pca)
print(f"ARI (KMeans on PCA {n_pcs} PCs): {ari_pca:.4f}")

Z_std = StandardScaler().fit_transform(Z)
km_vae = KMeans(n_clusters=K, n_init=100, random_state=SEED)
y_pred_vae = km_vae.fit_predict(Z_std)
ari_vae = adjusted_rand_score(y_true, y_pred_vae)
print(f"ARI (KMeans on VAE latent): {ari_vae:.4f}")