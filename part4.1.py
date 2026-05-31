"""
Part 4.1: Sampling Efficiency
- Train x-pred x-loss on all 3 datasets at D=32 (best model from Part 2)
- Save per-dataset checkpoints
- Evaluate sample quality across step counts: 1, 2, 5, 10, 20, 50, 100, 200
- Show how quality degrades as steps decrease
"""

import sys
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataloader import get_dataloader, ToyDiffusionDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

OUT_DIR = Path("part4_figures")
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# DETERMINISM
# ────────────────────────────────────────────────────────────────────────────

SEED = 13

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────────────────────
# MODEL (same as Parts 1-3)
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 2 == 0
        k = dim // 2
        i = torch.arange(k, dtype=torch.float32)
        freqs = torch.exp(-i * math.log(10000) / (k - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, None] * self.freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FlowMLP(nn.Module):
    def __init__(self, data_dim: int, hidden: int = 256, time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        in_dim = data_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        et = self.time_embed(t)
        x = torch.cat([z, et], dim=-1)
        return self.net(x)


# ────────────────────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────────────────────

EPS = 5e-2

def train(model, dataloader, n_steps=25_000, lr=1e-3, device=device):
    """x-pred x-loss training."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    data_iter = iter(dataloader)

    pbar = trange(n_steps, desc="Training x-pred x-loss D=32")
    for step in pbar:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        x = x.float().to(device)
        B = x.shape[0]

        t = torch.rand(B, device=device).clamp(EPS, 1 - EPS)
        eps = torch.randn_like(x)
        z_t = (1 - t[:, None]) * x + t[:, None] * eps

        x_pred = model(z_t, t)
        loss = nn.functional.mse_loss(x_pred, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return model


# ────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample(model, n_samples, data_dim, n_steps, device=device):
    """x-pred Euler sampling with variable step count."""
    model.eval()
    z = torch.randn(n_samples, data_dim, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for i in range(n_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = t.expand(n_samples)
        x_hat = model(z, t_batch)
        # convert x prediction to v for Euler step
        t_val = t.clamp(min=EPS)
        v_pred = (z - x_hat) / t_val
        z = z + v_pred * dt

    return z.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
# RUN: 3 DATASETS × 8 STEP COUNTS
# ────────────────────────────────────────────────────────────────────────────

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIM = 32
BATCH_SIZE = 1024
N_SAMPLES = 2000
STEP_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200]

for DATASET in DATASETS:
    print(f"\n{'='*60}\nDataset: {DATASET} D={DIM}\n{'='*60}")

    CHECKPOINT = OUT_DIR / f"part4_xpred_{DATASET}_d{DIM}.pt"

    # ── train or load ────────────────────────────────────────────────────────
    if CHECKPOINT.exists():
        print(f"Loading checkpoint: {CHECKPOINT}")
        model = FlowMLP(data_dim=DIM)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        model = model.to(device)
    else:
        print(f"Training x-pred x-loss on {DATASET} D={DIM}...")
        set_seed(SEED)
        dl = get_dataloader(name=DATASET, dim=DIM, batch_size=BATCH_SIZE)
        model = FlowMLP(data_dim=DIM)
        model = train(model, dl, n_steps=25_000, device=device)
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"Saved: {CHECKPOINT}")

    # ── ground truth for plotting ────────────────────────────────────────────
    ds = ToyDiffusionDataset(name=DATASET, dim=DIM)
    gt_2d = ds.to_2d(ds.data.numpy())

    # reseed so sampling is reproducible regardless of training path
    set_seed(SEED)

    # ── individual figures per step count ────────────────────────────────────
    for n_steps in STEP_COUNTS:
        print(f"  Sampling with {n_steps} steps...")
        samples = euler_sample(model, n_samples=N_SAMPLES, data_dim=DIM, n_steps=n_steps)
        samples_2d = ds.to_2d(samples)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"x-pred x-loss | {DATASET} D={DIM} | {n_steps} Euler steps",
                     fontsize=11)

        axes[0].scatter(gt_2d[:, 0], gt_2d[:, 1], s=1, alpha=0.5, color="orange")
        axes[0].set_title("ground truth")
        axes[0].set_aspect("equal")

        axes[1].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5,
                        color="steelblue")
        axes[1].set_title(f"generated ({n_steps} steps)")
        axes[1].set_aspect("equal")

        plt.tight_layout()
        fname = OUT_DIR / f"step_sweep_{DATASET}_{n_steps:03d}steps.png"
        plt.savefig(fname, dpi=120)
        plt.close()

    # ── summary grid for this dataset ────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Part 4.1 — Sampling Efficiency: x-pred x-loss {DATASET} D={DIM}",
                 fontsize=13)

    for idx, n_steps in enumerate(STEP_COUNTS):
        row, col = idx // 4, idx % 4
        samples = euler_sample(model, n_samples=N_SAMPLES, data_dim=DIM, n_steps=n_steps)
        samples_2d = ds.to_2d(samples)

        axes[row, col].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5,
                               color="steelblue")
        axes[row, col].set_title(f"{n_steps} steps")
        axes[row, col].set_aspect("equal")

    plt.tight_layout()
    fname = OUT_DIR / f"step_sweep_summary_{DATASET}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved summary: {fname}")

print("\nPart 4.1 complete (all 3 datasets).")
