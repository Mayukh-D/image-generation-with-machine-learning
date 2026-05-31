"""
Part 2: Flow Matching Parameterization
- 4 combinations: {x, v}-prediction x {x, v}-loss
- 3 datasets: swiss_roll, gaussians, circles
- 3 dimensions: D in {2, 8, 32}
- Total: 36 experiments
- All figures saved automatically, nothing blocks
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

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataloader import get_dataloader, ToyDiffusionDataset

# ── device ───────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── output folder ────────────────────────────────────────────────────────────
# Save all 36 figures here
OUT_DIR = Path("part2_figures")
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# MODEL (same as Part 1)
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
# CONVERSION FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────
# From forward process: z_t = (1-t)*x + t*eps
# Rearranging: x = z_t - t*v  (since v = eps - x, so z_t = x + t*v)
# And:         v = (z_t - x) / t

EPS = 5e-2

def x_to_v(x_hat, z_t, t):
    """Convert x prediction to v prediction.
    v_hat = (z_t - x_hat) / t
    t[:, None] broadcasts (B,) to (B, D)
    """
    return (z_t - x_hat) / t[:, None].clamp(min=EPS)

def v_to_x(v_hat, z_t, t):
    """Convert v prediction to x prediction.
    x_hat = z_t - t * v_hat
    """
    return z_t - t[:, None] * v_hat


# ────────────────────────────────────────────────────────────────────────────
# TRAINING - supports all 4 parameterizations
# ────────────────────────────────────────────────────────────────────────────

def train(
    model: FlowMLP,
    dataloader,
    pred_type: str,   # "x" or "v" - what the model outputs
    loss_type: str,   # "x" or "v" - what space the loss is computed in
    n_steps: int = 25_000,
    lr: float = 1e-3,
    device=device,
):
    """
    General training loop for all 4 parameterizations.

    pred_type: what the model's raw output represents
    loss_type: what we compute MSE against

    The 4 combinations:
    1. pred=x, loss=x: model outputs x_hat, MSE(x_hat, x)
    2. pred=x, loss=v: model outputs x_hat, convert to v_hat, MSE(v_hat, v)
    3. pred=v, loss=v: model outputs v_hat, MSE(v_hat, v)
    4. pred=v, loss=x: model outputs v_hat, convert to x_hat, MSE(x_hat, x)
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    data_iter = iter(dataloader)

    pbar = trange(n_steps, desc=f"pred={pred_type} loss={loss_type}", leave=False)

    for step in pbar:
        # ── get batch ────────────────────────────────────────────────────────
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        x = x.float().to(device)   # (B, D) clean data
        B = x.shape[0]

        # ── sample t and noise ───────────────────────────────────────────────
        t = torch.rand(B, device=device).clamp(EPS, 1 - EPS)
        eps = torch.randn_like(x)

        # ── forward process ──────────────────────────────────────────────────
        z_t = (1 - t[:, None]) * x + t[:, None] * eps   # noisy sample

        # ── true targets (always compute both, use whichever needed) ─────────
        v_true = eps - x     # true velocity
        x_true = x           # true clean data

        # ── model forward pass ───────────────────────────────────────────────
        # model output is always (B, D), interpretation depends on pred_type
        out = model(z_t, t)

        # ── compute loss based on pred_type and loss_type ────────────────────
        if pred_type == "x" and loss_type == "x":
            # model predicts x directly, loss in x space - no conversion needed
            loss = nn.functional.mse_loss(out, x_true)

        elif pred_type == "x" and loss_type == "v":
            # model predicts x, but loss in v space
            # convert predicted x to predicted v, then compare to true v
            v_hat = x_to_v(out, z_t, t)
            loss = nn.functional.mse_loss(v_hat, v_true)

        elif pred_type == "v" and loss_type == "v":
            # model predicts v directly, loss in v space - no conversion needed
            loss = nn.functional.mse_loss(out, v_true)

        elif pred_type == "v" and loss_type == "x":
            # model predicts v, but loss in x space
            # convert predicted v to predicted x, then compare to true x
            x_hat = v_to_x(out, z_t, t)
            loss = nn.functional.mse_loss(x_hat, x_true)

        # ── update ───────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")


# ────────────────────────────────────────────────────────────────────────────
# SAMPLING - handles both pred types
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample(
    model: FlowMLP,
    pred_type: str,     # "x" or "v" - what the model outputs
    n_samples: int,
    data_dim: int,
    n_steps: int = 50,
    device=device,
):
    """
    Euler ODE sampling.
    Regardless of pred_type, the ODE always steps using velocity v.
    If model predicts x, we convert to v first before stepping.
    """
    model.eval()
    z = torch.randn(n_samples, data_dim, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for i in range(n_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]   # negative
        t_batch = t.expand(n_samples)

        out = model(z, t_batch)   # (N, D)

        if pred_type == "x":
            # model predicted x_hat, convert to v_hat for the Euler step
            # v_hat = (z_t - x_hat) / t
            t_val = t.clamp(min=EPS)
            v_pred = (z - out) / t_val
        else:
            # model predicted v directly
            v_pred = out

        z = z + v_pred * dt

    return z.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
# SWEEP: 4 combos x 3 datasets x 3 dims = 36 experiments
# ────────────────────────────────────────────────────────────────────────────

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIMS = [2, 8, 32]
COMBOS = [
    ("x", "x"),   # x-prediction, x-loss
    ("x", "v"),   # x-prediction, v-loss
    ("v", "x"),   # v-prediction, x-loss
    ("v", "v"),   # v-prediction, v-loss
]

N_STEPS_TRAIN = 25_000
N_STEPS_SAMPLE = 50
BATCH_SIZE = 1024
LR = 1e-3
N_SAMPLES = 2000

total_runs = len(COMBOS) * len(DATASETS) * len(DIMS)
run = 0

for pred_type, loss_type in COMBOS:
    for dataset in DATASETS:
        for dim in DIMS:
            run += 1
            label = f"pred={pred_type}_loss={loss_type}_{dataset}_D{dim}"
            print(f"\n[{run}/{total_runs}] {label}")

            # ── train ────────────────────────────────────────────────────────
            dl = get_dataloader(name=dataset, dim=dim, batch_size=BATCH_SIZE)
            model = FlowMLP(data_dim=dim)
            train(model, dl, pred_type=pred_type, loss_type=loss_type,
                  n_steps=N_STEPS_TRAIN, lr=LR, device=device)

            # ── sample ───────────────────────────────────────────────────────
            samples = euler_sample(model, pred_type=pred_type,
                                   n_samples=N_SAMPLES, data_dim=dim,
                                   n_steps=N_STEPS_SAMPLE, device=device)

            # ── project to 2D for visualization ──────────────────────────────
            ds = ToyDiffusionDataset(name=dataset, dim=dim)
            gt = ds.data.numpy()

            # project both ground truth and samples to 2D if dim > 2
            if dim > 2:
                gt_2d = ds.to_2d(gt)
                samples_2d = ds.to_2d(samples)
            else:
                gt_2d = gt
                samples_2d = samples

            # ── plot and save ─────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.suptitle(f"pred={pred_type} loss={loss_type} | {dataset} D={dim}", fontsize=11)

            axes[0].scatter(gt_2d[:, 0], gt_2d[:, 1], s=1, alpha=0.5, color="orange")
            axes[0].set_title("ground truth")
            axes[0].set_aspect("equal")

            axes[1].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, color="steelblue")
            axes[1].set_title("generated")
            axes[1].set_aspect("equal")

            plt.tight_layout()

            # save figure - filename encodes all experiment details
            fname = OUT_DIR / f"{label}.png"
            plt.savefig(fname, dpi=120)
            plt.close()  # close immediately, never blocks
            print(f"  Saved: {fname}")

print(f"\nDone. All {total_runs} figures saved to {OUT_DIR}/")
