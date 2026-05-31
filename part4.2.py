"""
Part 4.2: MeanFlow - Final4.7
=============================

Hypothesis test: take final4 (Ideology 1) and add ONLY adaptive loss
weighting from final8 (Ideology 2). Leave logit-normal sampler OUT.

Completes the 2x2 ablation matrix:
  - final4    = neither trick     -> fuzzy swiss_roll, good gaussians
  - final4.5  = logit-normal only  -> fuzzy swiss_roll, gaussians collapse
  - final4.7  = adaptive weight only  (THIS FILE)
  - final8    = both tricks        -> crisp swiss_roll, gaussians collapse

If swiss_roll gets crisp and gaussians stays good, this is the sweet spot.
If gaussians collapses too, adaptive weighting alone is enough to kill
multi-modal density. If swiss_roll stays fuzzy, logit-normal was the
actual sharpener and we have no clean win.

Differences from final4:
  - Added adaptive_weight() helper applied to BOTH loss_fm and loss_mf
  - Everything else byte-identical: vmap+functional_call, uniform t,r,
    x-pred, flow_ratio=0.5, FM safety net.
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, jvp, functional_call
import matplotlib.pyplot as plt
from tqdm import trange

sys.path.insert(0, "/content/src")
from dataloader import get_dataloader, ToyDiffusionDataset

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

OUT_DIR = Path("/content/drive/MyDrive/part4_final47")
OUT_DIR.mkdir(exist_ok=True, parents=True)

EPS = 1e-3

# ────────────────────────────────────────────────────────────────────────────
# MODEL
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        k = dim // 2
        i = torch.arange(k, dtype=torch.float32)
        freqs = torch.exp(-i * math.log(10000) / (k - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(t.device)  # safe inside vmap
        angles = t[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class MeanFlowMLP(nn.Module):
    """
    MeanFlow model. Predicts x_hat given (z_t, t, h=t-r).
    Two separate sinusoidal embeddings per assignment spec.
    Input: [z_t; e_t; e_h] in R^{D+256}
    """
    def __init__(self, data_dim: int, hidden: int = 256, time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)      # for t
        self.horizon_embed = SinusoidalEmbedding(time_dim)   # for h, separate params
        in_dim = data_dim + 2 * time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, z, t, h):
        et = self.time_embed(t)
        eh = self.horizon_embed(h)
        return self.net(torch.cat([z, et, eh], dim=-1))


# ────────────────────────────────────────────────────────────────────────────
# ADAPTIVE LOSS WEIGHTING  (the only addition vs final4)
# ────────────────────────────────────────────────────────────────────────────

def adaptive_weight(loss_per_sample, norm_p=1.0, norm_eps=0.01):
    """
    From MeanFlow paper Section 4.3, Table 1e (norm_p=1.0 optimal).
        weight = stopgrad((loss + eps)^p)
        loss_weighted = loss / weight
    Downweights samples with high loss to keep gradients balanced.
    """
    with torch.no_grad():
        weight = (loss_per_sample + norm_eps) ** norm_p
    return loss_per_sample / weight


# ────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ────────────────────────────────────────────────────────────────────────────

def train_step(model, x, optimizer, device, flow_ratio=0.5):
    B, D = x.shape

    # sample t uniform in [EPS, 1-EPS]
    t = torch.rand(B, device=device).clamp(EPS, 1 - EPS)

    # sample r uniform in [EPS, t-EPS] - clean, no distribution distortion
    r = (torch.rand_like(t) * (t - EPS)).clamp(min=EPS)
    h = (t - r).clamp(min=EPS)

    # forward process
    eps = torch.randn_like(x)
    z_t = (1 - t[:, None]) * x + t[:, None] * eps
    v_true = eps - x

    # ── Flow Matching loss (h=0, x-pred x-loss) ──────────────────────────────
    x_hat_fm = model(z_t, t, torch.zeros_like(t))
    loss_fm = ((x_hat_fm - x) ** 2).mean(dim=1)   # (B,)
    loss_fm = adaptive_weight(loss_fm)

    # ── MeanFlow loss (h>0) ───────────────────────────────────────────────────
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def fn_single(z_i, r_i, t_i):
        # compute h_i = t_i - r_i inside JVP
        h_i = (t_i - r_i).clamp(min=EPS)
        # use functional_call to safely call model inside vmap
        x_hat = functional_call(
            model, {**params, **buffers},
            (z_i.unsqueeze(0), t_i.unsqueeze(0), h_i.unsqueeze(0))
        ).squeeze(0)
        # return u_hat = (z - x_hat) / t (average velocity from x-pred)
        return (z_i - x_hat) / t_i.clamp(min=EPS)

    def jvp_single(z_i, r_i, t_i, v_i):
        # tangent (v, 0, 1) matches paper Algorithm 1 exactly
        return jvp(
            fn_single,
            (z_i, r_i, t_i),
            (v_i, torch.zeros_like(r_i), torch.ones_like(t_i))
        )

    # per-sample JVP via vmap - no cross-sample coupling
    u_hat, dudt = vmap(jvp_single)(z_t, r, t, v_true)

    # MeanFlow target: u_tgt = v - (t-r) * dudt, stopgrad
    u_tgt = (v_true - h[:, None] * dudt).detach()
    loss_mf = ((u_hat - u_tgt) ** 2).mean(dim=1)
    loss_mf = adaptive_weight(loss_mf)

    # per-sample mix of FM and MeanFlow
    mask = torch.rand(B, device=device) < flow_ratio
    loss = torch.where(mask, loss_fm, loss_mf).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ────────────────────────────────────────────────────────────────────────────

def train_meanflow(model, dataloader, n_steps=50_000, lr=1e-3, device=device):
    model = model.to(device)  # critical: move model to device first
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_iter = iter(dataloader)

    pbar = trange(n_steps, desc="MeanFlow training")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.float().to(device)

        loss = train_step(model, x, optimizer, device)

        if step % 2000 == 0:
            pbar.set_postfix(loss=f"{loss:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def meanflow_sample(model, n_samples, data_dim, n_steps=1, device=device):
    """
    z_r = z_t - (t-r) * u(z_t, r, t)
    where u = (z - x_hat) / t from x-prediction
    1-step: z0 = z1 - u(z1, r=0, t=1)
    """
    model.eval()
    z = torch.randn(n_samples, data_dim, device=device)
    ts = torch.linspace(1.0, EPS, n_steps + 1, device=device)

    for i in range(n_steps):
        t_val = ts[i]
        r_val = ts[i + 1]
        h_val = (t_val - r_val).clamp(min=EPS)

        x_hat = model(z,
                      t_val.expand(n_samples),
                      h_val.expand(n_samples))
        u = (z - x_hat) / t_val.clamp(min=EPS)
        z = z - h_val * u

    return z.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
# RUN: 3 datasets x D=32 x {1,2,5} steps = 9 figures
# ────────────────────────────────────────────────────────────────────────────

DATASETS = ["swiss_roll", "gaussians", "circles"]
SEED_MAP = {"swiss_roll": 0,"gaussians": 100, "circles": 456}
DIM = 32
BATCH_SIZE = 1024
N_STEPS_TRAIN = 50_000
N_SAMPLES = 2000
MEANFLOW_STEPS = [1, 2, 5]

for dataset in DATASETS:
    seed = SEED_MAP[dataset]
    set_seed(seed)
    print(f"\n{'='*50}\nDataset: {dataset} D={DIM}")

    ckpt_path = OUT_DIR / f"meanflow_{dataset}_D{DIM}_final47.pt"
    dl = get_dataloader(name=dataset, dim=DIM, batch_size=BATCH_SIZE)
    ds = ToyDiffusionDataset(name=dataset, dim=DIM)
    gt_2d = ds.to_2d(ds.data.numpy())

    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        model = MeanFlowMLP(data_dim=DIM)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
    else:
        model = MeanFlowMLP(data_dim=DIM)
        train_meanflow(model, dl, n_steps=N_STEPS_TRAIN, device=device)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

    for n_steps in MEANFLOW_STEPS:
        print(f"  Sampling {n_steps} step(s)...")
        samples = meanflow_sample(model, N_SAMPLES, DIM, n_steps, device)
        samples_2d = ds.to_2d(samples)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"MeanFlow | {dataset} D={DIM} | {n_steps} step(s)", fontsize=11)
        axes[0].scatter(gt_2d[:, 0], gt_2d[:, 1], s=1, alpha=0.5, color="orange")
        axes[0].set_title("ground truth")
        axes[0].set_aspect("equal")
        axes[1].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, color="steelblue")
        axes[1].set_title(f"{n_steps} step(s)")
        axes[1].set_aspect("equal")
        plt.tight_layout()
        fname = OUT_DIR / f"meanflow_{dataset}_D{DIM}_{n_steps}steps_final47.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved: {fname}")

print(f"\nDone. 9 figures in {OUT_DIR}/")
