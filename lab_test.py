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
import os, random, torch

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Info] Using device:", device)


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

gamma = 3     # main knob
rng = np.ptp(disp) if np.ptp(disp) != 0 else 1.0  # calculates range for each gene
w = 1.0 + gamma * (disp - disp.min()) / (rng + 1e-4)  # weights >= 1 for more variable genes
w = w.astype(np.float32)

# Keep overall loss scale comparable to unweighted MSE (avg weight ~ 1)
w *= (w.size / w.sum())


# --------- VAE MODEL ---------
X = np.asarray(adata.X, dtype=np.float32)
assert w.shape[0] == X.shape[1]
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
            nn.Linear(input_dim, 256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU()
        )
        self.mu = nn.Linear(64, latent_dim)       
        self.logvar = nn.Linear(64, latent_dim)        
        self.dec = nn.Sequential(       # Mirrors the encoder but back to the input space
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 256), nn.GELU(),
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

# Labels and ARI Calculator
label_col = "label"
labels = adata.obs[label_col].astype("category")
y_true = labels.cat.codes.to_numpy()
K = len(labels.cat.categories)
eval_dl = tud.DataLoader(ds, batch_size=1024, shuffle=False)

def ari_from_model(model, eval_dl, y_true, K, seed=SEED):
    model.eval()                           # <- parentheses!
    mu_chunks = []
    with torch.no_grad():
        for (xb,) in eval_dl:
            xb = xb.to(device)
            mu, _ = model.encode(xb)       # use μ (deterministic)
            mu_chunks.append(mu.cpu().numpy())
    Z = np.concatenate(mu_chunks, axis=0)  # (n_cells, d_latent)
    Z_std = StandardScaler().fit_transform(Z)
    y_pred = KMeans(n_clusters=K, n_init=100, random_state=seed).fit_predict(Z_std)
    ari = adjusted_rand_score(y_true, y_pred)
    return ari, Z

def ari_from_array(X_rep, y_true, K, seed=SEED):
    X_std = StandardScaler().fit_transform(X_rep)
    y_pred = KMeans(n_clusters=K, n_init=100, random_state=seed).fit_predict(X_std)
    return adjusted_rand_score(y_true, y_pred)

# Training
d_in, d_latent = X.shape[1], 2         # input features and latent dimensionality for VAE
device = "cpu"                                     
model = VAE(d_in, d_latent).to(device)          # moves parameters to evice
opt = torch.optim.Adam(model.parameters(), lr=1e-3)         # solid default for Adam optimizer

# --- Make a torch tensor for the per-gene weights ---
gene_w = torch.from_numpy(w).to(device)   # shape (d_in,)

epochs = 50
warmup = 20

print(f"[Info] Training on {device} for {epochs} epochs...")
for epoch in range(1, epochs + 1):
    model.train()
    total = total_recon = total_kl = 0.0
    for (xb,) in dl:
        xb = xb.to(device)
        opt.zero_grad()
        x_hat, mu, logvar = model(xb)
        loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, gene_w=gene_w)

        beta = min(1.0, epoch / warmup)          # <- warm-up (optional)
        (recon + beta * kl).backward()           # instead of loss.backward()
        opt.step()

        total += (recon + kl).item()
        total_recon += recon.item()
        total_kl += kl.item()

    ari_epoch, _ = ari_from_model(model, eval_dl, y_true, K, seed=SEED)
    print(f"Epoch {epoch:03d} | loss/cell={total/len(ds):.2f} "
          f"(recon/cell={total_recon/len(ds):.2f}, kl/cell={total_kl/len(ds):.2f}) "
          f"| ARI={ari_epoch:.4f}")
    
# --------- PCA and ARI alculation ---------
n_pcs = 16 
sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=SEED)
X_pca = adata.obsm["X_pca"][:, :n_pcs]        # (n_cells, n_pcs)

ari_vae_final, Z = ari_from_model(model, eval_dl, y_true, K, seed=SEED)
ari_pca_final    = ari_from_array(X_pca, y_true, K, seed=SEED)
print(f"[Final] ARI VAE: {ari_vae_final:.4f} | ARI PCA({n_pcs}): {ari_pca_final:.4f}")

# --------- UMAP ---------
def umap_embed(X, seed=SEED):
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                     random_state=seed).fit_transform(X)

#emb_vae = umap_embed(Z, SEED)
#emb_pca = umap_embed(X_pca, SEED)