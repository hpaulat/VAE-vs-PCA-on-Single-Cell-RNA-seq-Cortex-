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
import os, random

SEED = 0            # seeding
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
adata.layers["counts"] = adata.X.copy()             # store counts in layers
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# --------- FEATURE SELECTION ---------
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)           # subset for analysis
print("[HVG] using", adata.n_vars, "genes")

# --------- WEIGHT MATRIX ---------
disp = adata.var["dispersions_norm"].to_numpy()             # dispersions extracted to NumPy array
disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)      # sets low quality values to 0 (float32) 

gamma = 2     # main knob
rng = np.ptp(disp) if np.ptp(disp) != 0 else 1.0             # range of array
w = 1.0 + gamma * (disp - disp.min()) / (rng + 1e-4)               # weight matrix from dispersion values
w = w.astype(np.float32)
w *= (w.size / w.sum())             # normalizes avg weight to 1, comparable to unweighted MSE

# --------- VAE MODEL ---------
X = np.asarray(adata.X, dtype=np.float32)
assert w.shape[0] == X.shape[1]             # check weight matrix same size
print(f"[Check] X dimensions = {X.shape}, dtype = {X.dtype}, type = {type(X)}")

batch_size = 256
ds = tud.TensorDataset(torch.from_numpy(X))
g = torch.Generator(device=device).manual_seed(SEED)        
dl = tud.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)              # Data Loader
print(f"[Sanity] n_cells = {len(ds)}, n_genes = {X.shape[1]},"
      f"batches/epoch ≈ {math.ceil(len(ds)/batch_size)}")

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):  
        super().__init__()
        self.enc = nn.Sequential(       # Two fully connected layers, GeLU AF
            nn.Linear(input_dim, 1024), nn.GELU(), nn.LayerNorm(1024),
            nn.Linear(1024, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), 
        )
        self.mu = nn.Linear(64, latent_dim)       
        self.logvar = nn.Linear(64, latent_dim)        
        self.dec = nn.Sequential(       # Mirrors encoder
            nn.Linear(latent_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 1024), nn.GELU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
    
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)               # convert logvar to std
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):              # forward pass (encode, reparametrize trick, decode)
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, gene_w, kl_weight = 0.8):            # beta_VAE_loss
    diff2 = (x_hat - x) ** 2                # per-element squared error
    recon = (diff2 * gene_w).sum() if gene_w is not None else diff2.sum()       #up/down weights genes by importance
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum()               # closed form KL Divergence summed over batch and latent dims

    B = x.size(0)
    recon = recon / B           # Keeps losses at same scale regardless of batch size
    kl = kl / B

    total = recon + kl_weight * kl
    return total, recon, kl

label_col = "label"
labels = adata.obs[label_col].astype("category")
y_true = labels.cat.codes.to_numpy()             # array of proper labels
K = len(labels.cat.categories)              # number of expected batches
eval_dl = tud.DataLoader(ds, batch_size=1024, shuffle=False)

def ari_from_model(model, eval_dl, y_true, K, seed=SEED):
    model.eval()                           
    mu_chunks = []
    with torch.no_grad():
        for (xb,) in eval_dl:
            xb = xb.to(device)
            mu, _ = model.encode(xb)               # collecting μ for all batches
            mu_chunks.append(mu.cpu().numpy())
    Z = np.concatenate(mu_chunks, axis=0)
    Z_std = StandardScaler().fit_transform(Z)
    y_pred = KMeans(n_clusters=K, n_init=100, random_state=seed).fit_predict(Z_std)
    ari = adjusted_rand_score(y_true, y_pred)
    return ari, Z

def ari_from_array(X_rep, y_true, K, seed=SEED):
    X_std = StandardScaler().fit_transform(X_rep)
    y_pred = KMeans(n_clusters=K, n_init=100, random_state=seed).fit_predict(X_std)
    return adjusted_rand_score(y_true, y_pred)

# Training
d_in, d_latent = X.shape[1], 6         # input features and latent dimensionality for VAE
model = VAE(d_in, d_latent).to(device)          # moves parameters to device
opt = torch.optim.Adam(model.parameters(), lr=1e-3)         # solid default for Adam optimizer
gene_w = torch.from_numpy(w).to(device)

epochs = 50
warmup = 40

def beta_cosine(epoch, warmup, beta_max=0.1):
    if epoch >= warmup: 
        return beta_max
    return beta_max * 0.5 * (1 - math.cos(math.pi * epoch / warmup))

print(f"[Info] Training on {device} for {epochs} epochs...")
for epoch in range(1, epochs + 1):
    model.train()
    total = total_recon = total_kl = 0.0
    beta = beta_cosine(epoch, warmup=warmup, beta_max=0.1)   

    for (xb,) in dl:
        xb = xb.to(device)
        opt.zero_grad()
        x_hat, mu, logvar = model(xb)
        loss_sum, recon_sum, kl_sum = vae_loss(xb, x_hat, mu, logvar, gene_w=gene_w, kl_weight=beta)
        
        B = xb.size(0)
        loss_sum.backward()
        opt.step()

        total       += (recon_sum.item() + beta * kl_sum.item()) * B
        total_recon +=  recon_sum.item() * B
        total_kl    +=  kl_sum.item() * B
    ari_epoch, _ = ari_from_model(model, eval_dl, y_true, K, seed=SEED)

    print(f"Epoch {epoch:03d} | loss/cell={total/len(ds):.2f} "
          f"(recon/cell={total_recon/len(ds):.2f}, kl/cell={total_kl/len(ds):.2f}) "
          f"| ARI={ari_epoch:.4f}")
    

n_pcs = 16            # PCA and ARI
sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=SEED)
X_pca = adata.obsm["X_pca"][:, :n_pcs]        # (n_cells, n_pcs)

ari_vae_final, Z = ari_from_model(model, eval_dl, y_true, K, seed=SEED)
ari_pca_final    = ari_from_array(X_pca, y_true, K, seed=SEED)
print(f"[Final] ARI VAE: {ari_vae_final:.4f} | ARI PCA({n_pcs}): {ari_pca_final:.4f}")


# --------- UMAP ---------
def umap_embed(X, seed=SEED):
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                     random_state=seed).fit_transform(X)

emb_vae = umap_embed(Z, SEED)
emb_pca = umap_embed(X_pca, SEED)
adata.obsm["X_umap_vae"] = emb_vae
adata.obsm["X_umap_pca"] = emb_pca

label_names = list(labels.cat.categories)
K = len(label_names)

cmap = plt.colormaps.get_cmap("tab20")
point_colors = cmap(y_true) 

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)

axes[0].scatter(emb_vae[:, 0], emb_vae[:, 1], c=point_colors, s=6, alpha=0.9, linewidths=0)
axes[0].set_title("UMAP of VAE μ")
axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].scatter(emb_pca[:, 0], emb_pca[:, 1], c=point_colors, s=6, alpha=0.9, linewidths=0)
axes[1].set_title("UMAP of PCA")
axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")
axes[1].set_xticks([]); axes[1].set_yticks([])

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label=label_names[i],
           markerfacecolor=cmap(i), markersize=6)
    for i in range(K)
]
axes[1].legend(
    legend_handles, [h.get_label() for h in legend_handles],
    title="Cell type", loc="upper left", bbox_to_anchor=(1.02, 1.0),
    frameon=False, fontsize=8, ncol=1  # bump ncol if you have many labels
)
plt.show()
