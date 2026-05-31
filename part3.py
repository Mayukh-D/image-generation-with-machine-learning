"""
Part 3: Can We Rescue v-Prediction?

Based on RAE Section 4:
- 4.1: Model width must match or exceed data dimension (Theorem 1)
- 4.2: Dimension-dependent noise schedule shift

Experiments (all on swiss_roll D=32, v-pred v-loss):
- Baseline:  hidden=256, no shift      (already failing from Part 2)
- Exp 1:     hidden=512, no shift
- Exp 2:     hidden=1024, no shift
- Exp 3:     hidden=256, with shift
- Exp 4:     hidden=512, with shift
- Exp 5:     hidden=1024, with shift
+ x-pred baseline for comparison
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataloader import get_dataloader, ToyDiffusionDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

OUT_DIR = Path("part3_figures")
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# MODEL
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
# DIMENSION-DEPENDENT NOISE SCHEDULE SHIFT (RAE Section 4.2)
# ────────────────────────────────────────────────────────────────────────────

def shift_t(t: torch.Tensor, data_dim: int, base_dim: int = 4096) -> torch.Tensor:
    """
    Apply dimension-dependent noise schedule shift from RAE Section 4.2.

    At high dimensions, Gaussian noise corrupts the signal less per dimension.
    This shift compensates by effectively injecting more noise at the same t.

    Formula: t_m = (alpha * t_n) / (1 + (alpha - 1) * t_n)
    where alpha = sqrt(m / n), m = effective data dim, n = base dim (4096)

    Args:
        t: raw timesteps in [0, 1]
        data_dim: actual data dimension (m)
        base_dim: reference base dimension (n=4096 per RAE paper)
    """
    alpha = math.sqrt(data_dim / base_dim)
    return (alpha * t) / (1 + (alpha - 1) * t)


# ────────────────────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────────────────────

EPS = 5e-2

def train(
    model: FlowMLP,
    dataloader,
    pred_type: str,
    loss_type: str,
    data_dim: int,
    use_schedule_shift: bool = False,
    n_steps: int = 25_000,
    lr: float = 1e-3,
    device=device,
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    data_iter = iter(dataloader)

    pbar = trange(n_steps, desc=f"pred={pred_type} loss={loss_type} shift={use_schedule_shift}", leave=False)

    for step in pbar:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        x = x.float().to(device)
        B = x.shape[0]

        # sample t
        t = torch.rand(B, device=device).clamp(EPS, 1 - EPS)

        # apply schedule shift if requested (RAE 4.2)
        if use_schedule_shift:
            t = shift_t(t, data_dim=data_dim)

        eps = torch.randn_like(x)
        z_t = (1 - t[:, None]) * x + t[:, None] * eps

        v_true = eps - x
        x_true = x

        out = model(z_t, t)

        if pred_type == "x" and loss_type == "x":
            loss = nn.functional.mse_loss(out, x_true)
        elif pred_type == "x" and loss_type == "v":
            v_hat = (z_t - out) / t[:, None].clamp(min=EPS)
            loss = nn.functional.mse_loss(v_hat, v_true)
        elif pred_type == "v" and loss_type == "v":
            loss = nn.functional.mse_loss(out, v_true)
        elif pred_type == "v" and loss_type == "x":
            x_hat = z_t - t[:, None] * out
            loss = nn.functional.mse_loss(x_hat, x_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")


# ────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample(model, pred_type, n_samples, data_dim, n_steps=50, device=device):
    model.eval()
    z = torch.randn(n_samples, data_dim, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for i in range(n_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = t.expand(n_samples)
        out = model(z, t_batch)

        if pred_type == "x":
            t_val = t.clamp(min=EPS)
            v_pred = (z - out) / t_val
        else:
            v_pred = out

        z = z + v_pred * dt

    return z.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ────────────────────────────────────────────────────────────────────────────

DATASET = "swiss_roll"
DIM = 32
BATCH_SIZE = 1024
N_STEPS_TRAIN = 25_000
N_STEPS_SAMPLE = 50
N_SAMPLES = 2000

# load ground truth once
ds = ToyDiffusionDataset(name=DATASET, dim=DIM)
gt_2d = ds.to_2d(ds.data.numpy())

def run_experiment(name, pred_type, loss_type, hidden, use_schedule_shift):
    print(f"\n[{name}] pred={pred_type} loss={loss_type} hidden={hidden} shift={use_schedule_shift}")

    dl = get_dataloader(name=DATASET, dim=DIM, batch_size=BATCH_SIZE)
    model = FlowMLP(data_dim=DIM, hidden=hidden)

    train(model, dl,
          pred_type=pred_type,
          loss_type=loss_type,
          data_dim=DIM,
          use_schedule_shift=use_schedule_shift,
          n_steps=N_STEPS_TRAIN,
          device=device)

    samples = euler_sample(model, pred_type=pred_type,
                           n_samples=N_SAMPLES, data_dim=DIM,
                           n_steps=N_STEPS_SAMPLE, device=device)
    samples_2d = ds.to_2d(samples)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(name, fontsize=11)

    axes[0].scatter(gt_2d[:, 0], gt_2d[:, 1], s=1, alpha=0.5, color="orange")
    axes[0].set_title("ground truth")
    axes[0].set_aspect("equal")

    axes[1].scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, color="steelblue")
    axes[1].set_title("generated")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    fname = OUT_DIR / f"{name}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")


# x-pred baseline (should work, comparison reference)
run_experiment("xpred_baseline_h256",
               pred_type="x", loss_type="x", hidden=256, use_schedule_shift=False)

# v-pred baseline (fails, already seen in Part 2)
run_experiment("vpred_baseline_h256",
               pred_type="v", loss_type="v", hidden=256, use_schedule_shift=False)

# RAE 4.1: increase width only
run_experiment("vpred_h512_noshift",
               pred_type="v", loss_type="v", hidden=512, use_schedule_shift=False)

run_experiment("vpred_h1024_noshift",
               pred_type="v", loss_type="v", hidden=1024, use_schedule_shift=False)

# RAE 4.2: schedule shift only
run_experiment("vpred_h256_shift",
               pred_type="v", loss_type="v", hidden=256, use_schedule_shift=True)

# Combined: width + shift
run_experiment("vpred_h512_shift",
               pred_type="v", loss_type="v", hidden=512, use_schedule_shift=True)

run_experiment("vpred_h1024_shift",
               pred_type="v", loss_type="v", hidden=1024, use_schedule_shift=True)

print(f"\nDone. All figures saved to {OUT_DIR}/")
