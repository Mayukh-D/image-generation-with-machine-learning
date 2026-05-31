"""
Part 1: Warm-up
- Section 3.1: Data Visualization (6 figures)
- Section 3.2: v-prediction flow matching at D=2
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import trange

# ── make sure src/ is on the path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataloader import get_dataloader, ToyDiffusionDataset

# ── device ──────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────────────────────────────────
# 3.1  DATA VISUALIZATION
# ────────────────────────────────────────────────────────────────────────────

DATASETS = ["swiss_roll", "gaussians", "circles"]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Part 1 — Data Visualization", fontsize=14)

for col, name in enumerate(DATASETS):
    # D=2 original
    ds2 = ToyDiffusionDataset(name=name, dim=2)
    data2 = ds2.data.numpy()
    axes[0, col].scatter(data2[:, 0], data2[:, 1], s=1, alpha=0.5)
    axes[0, col].set_title(f"{name} D=2")
    axes[0, col].set_aspect("equal")

    # D=32 projected back to 2D
    ds32 = ToyDiffusionDataset(name=name, dim=32)
    data32_2d = ds32.to_2d(ds32.data.numpy())
    axes[1, col].scatter(data32_2d[:, 0], data32_2d[:, 1], s=1, alpha=0.5, color="orange")
    axes[1, col].set_title(f"{name} D=32 → 2D")
    axes[1, col].set_aspect("equal")

plt.tight_layout()
plt.savefig("part1_data_visualization.png", dpi=150)
plt.show()
print("Saved: part1_data_visualization.png")

# ────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Fixed sinusoidal time embedding (DiT-style).
    Maps scalar t -> R^128
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 2 == 0
        k = dim // 2
        # frequencies: shape (k,)
        i = torch.arange(k, dtype=torch.float32)
        freqs = torch.exp(-i * math.log(10000) / (k - 1))
        self.register_buffer("freqs", freqs)  # fixed, not trained

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar times
        # freqs: (k,)
        angles = t[:, None] * self.freqs[None, :]   # (B, k)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, 128)


class FlowMLP(nn.Module):
    """
    MLP denoiser for flow matching.
    Input:  [z_t; e_t] in R^{D+128}
    Output: prediction in R^D  (interpretation depends on pred_type)
    """
    def __init__(self, data_dim: int, hidden: int = 256, time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        in_dim = data_dim + time_dim

        # 5 hidden layers + 1 output layer = 6 Linear layers total
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, data_dim),   # no activation
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: (B, D),  t: (B,)
        et = self.time_embed(t)          # (B, 128)
        x = torch.cat([z, et], dim=-1)  # (B, D+128)
        return self.net(x)               # (B, D)


# ────────────────────────────────────────────────────────────────────────────
# TRAINING AND SAMPLING
# ────────────────────────────────────────────────────────────────────────────

EPS = 1e-4  # clip t away from 0 and 1 for numerical stability

def train_flow_matching(
    model: FlowMLP,
    dataloader,
    n_steps: int = 25_000,
    lr: float = 1e-3,
    device=device,
):
    """
    v-prediction with v-loss training loop.
    Forward process: z_t = (1-t)*x + t*eps
    Target: v = eps - x
    Loss: MSE(model(z_t, t), v)
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    data_iter = iter(dataloader)

    losses = []
    pbar = trange(n_steps, desc="Training")
    for step in pbar:
        # get batch
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        x = x.float().to(device)           # (B, D)
        B = x.shape[0]

        # sample t and eps
        t = torch.rand(B, device=device).clamp(EPS, 1 - EPS)  # (B,)
        eps = torch.randn_like(x)                               # (B, D)

        # forward process
        z_t = (1 - t[:, None]) * x + t[:, None] * eps         # (B, D)

        # target velocity
        v = eps - x                                             # (B, D)

        # model predicts v directly (v-prediction)
        v_pred = model(z_t, t)                                  # (B, D)

        loss = nn.functional.mse_loss(v_pred, v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 2000 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return losses


@torch.no_grad()
def euler_sample(
    model: FlowMLP,
    n_samples: int,
    data_dim: int,
    n_steps: int = 50,
    device=device,
):
    """
    Euler ODE sampling for v-prediction model.
    Start from z ~ N(0,I) at t=1, step toward t=0.
    """
    model.eval()
    z = torch.randn(n_samples, data_dim, device=device)  # start at t=1

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)  # t=1 down to t=0

    for i in range(n_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]  # negative (stepping toward 0)
        t_batch = t.expand(n_samples)
        v_pred = model(z, t_batch)
        z = z + v_pred * dt

    return z.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
# 3.2  v-PREDICTION AT D=2 FOR ALL 3 DATASETS
# ────────────────────────────────────────────────────────────────────────────

N_STEPS_TRAIN = 25_000
N_STEPS_SAMPLE = 50
BATCH_SIZE = 1024
LR = 1e-3
DIM = 2

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Part 1 — v-prediction Flow Matching at D=2", fontsize=14)

for col, name in enumerate(DATASETS):
    print(f"\nTraining on {name} D={DIM}...")

    # ground truth
    ds = ToyDiffusionDataset(name=name, dim=DIM)
    gt = ds.data.numpy()
    axes[0, col].scatter(gt[:, 0], gt[:, 1], s=1, alpha=0.5, color="orange")
    axes[0, col].set_title(f"{name} — ground truth")
    axes[0, col].set_aspect("equal")

    # train
    dl = get_dataloader(name=name, dim=DIM, batch_size=BATCH_SIZE)
    model = FlowMLP(data_dim=DIM)
    losses = train_flow_matching(model, dl, n_steps=N_STEPS_TRAIN, lr=LR)

    # sample
    samples = euler_sample(model, n_samples=2000, data_dim=DIM, n_steps=N_STEPS_SAMPLE)
    axes[1, col].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, color="steelblue")
    axes[1, col].set_title(f"{name} — generated")
    axes[1, col].set_aspect("equal")

plt.tight_layout()
plt.savefig("part1_vpred_d2.png", dpi=150)
plt.show()
print("\nSaved: part1_vpred_d2.png")
print("Part 1 complete.")
