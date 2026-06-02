"""Minimal, self-contained 2D DDPM (Denoising Diffusion Probabilistic Model).

This is the plain-script twin of notebook 02. It trains a tiny DDPM on the
sklearn `make_moons` 2D point cloud and then generates new points by reversing
the diffusion (ancestral sampling). It is meant to be the smallest end-to-end
DDPM you can read top-to-bottom in one sitting:

  data  ->  forward noising q(x_t | x_0)  ->  learn eps_theta(x_t, t)
        ->  reverse ancestral sampling p(x_{t-1} | x_t)  ->  new samples

What it does, step by step:
  1. Builds a make_moons dataset (2D, two interleaving half-circles).
  2. Defines a small MLP `eps_theta` that predicts the added noise, conditioned
     on the timestep via a sinusoidal time embedding.
  3. Uses a linear beta schedule with T = 200 diffusion steps.
  4. Trains with the L_simple objective (MSE between true and predicted noise)
     for ~2000 Adam steps, batch size 256.
  5. Prints the final loss.
  6. Ancestral-samples ~2000 points from pure noise back to data.
  7. Saves a side-by-side real-vs-generated scatter plot to samples.png next to
     this script.

Runs headless on CPU in well under a minute. No GPU, no plt.show().

HOW TO RUN (from this examples directory):

    uv run python ddpm_min.py

Output: writes `samples.png` next to this script.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed, just savefig
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# Reproducibility.
torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data: make_moons, standardized to roughly unit scale.
# --------------------------------------------------------------------------- #
def build_data(n=4000):
    x, _ = make_moons(n_samples=n, noise=0.05, random_state=0)
    x = x.astype(np.float32)
    # Standardize each coordinate so the cloud is centered near 0 with unit-ish
    # spread -- keeps it compatible with the standard-normal diffusion prior.
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return torch.from_numpy(x)


# --------------------------------------------------------------------------- #
# Sinusoidal time embedding (Transformer-style positional encoding of t).
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) integer or float timesteps.
        half = self.dim // 2
        # Geometric range of frequencies.
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:  # pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb  # (B, dim)


# --------------------------------------------------------------------------- #
# eps_theta: small MLP predicting the noise added at step t.
# --------------------------------------------------------------------------- #
class EpsTheta(nn.Module):
    def __init__(self, data_dim=2, time_dim=32, hidden=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t):
        temb = self.time_mlp(self.time_embed(t))
        h = torch.cat([x, temb], dim=1)
        return self.net(h)


# --------------------------------------------------------------------------- #
# Diffusion constants: linear beta schedule and derived alphas.
# --------------------------------------------------------------------------- #
def make_schedule(T=200, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(beta_start, beta_end, T, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_acp": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_acp": torch.sqrt(1.0 - alphas_cumprod),
    }


# --------------------------------------------------------------------------- #
# Training: L_simple = E || eps - eps_theta(x_t, t) ||^2.
# --------------------------------------------------------------------------- #
def train(model, data, sched, steps=2000, batch=256, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = sched["T"]
    n = data.shape[0]
    model.train()
    final_loss = float("nan")
    for step in range(steps):
        idx = torch.randint(0, n, (batch,))
        x0 = data[idx]  # (B, 2)
        t = torch.randint(0, T, (batch,), device=DEVICE)  # (B,)
        noise = torch.randn_like(x0)
        # Forward process closed form: x_t = sqrt(acp) x0 + sqrt(1-acp) eps.
        sqrt_acp = sched["sqrt_acp"][t][:, None]
        sqrt_omacp = sched["sqrt_one_minus_acp"][t][:, None]
        x_t = sqrt_acp * x0 + sqrt_omacp * noise
        pred = model(x_t, t)
        loss = ((noise - pred) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return final_loss


# --------------------------------------------------------------------------- #
# Reverse process: ancestral sampling from x_T ~ N(0, I) down to x_0.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model, sched, n_samples=2000):
    model.eval()
    T = sched["T"]
    betas = sched["betas"]
    alphas = sched["alphas"]
    acp = sched["alphas_cumprod"]
    x = torch.randn(n_samples, 2, device=DEVICE)
    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
        eps = model(x, t)
        beta_t = betas[i]
        alpha_t = alphas[i]
        acp_t = acp[i]
        # Posterior mean of x_{t-1}.
        coef = beta_t / torch.sqrt(1.0 - acp_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * eps)
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return x


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "samples.png")

    data = build_data(n=4000)
    sched = make_schedule(T=200)
    model = EpsTheta().to(DEVICE)

    final_loss = train(model, data, sched, steps=2000, batch=256, lr=1e-3)
    print("final loss: %.4f" % final_loss)

    gen = sample(model, sched, n_samples=2000).cpu().numpy()
    real = data.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(real[:, 0], real[:, 1], s=4, alpha=0.5, color="tab:blue")
    ax1.set_title("real (make_moons)")
    ax1.set_aspect("equal")
    ax2.scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.5, color="tab:orange")
    ax2.set_title("generated (DDPM)")
    ax2.set_aspect("equal")
    for ax in (ax1, ax2):
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    fig.suptitle("2D DDPM: real vs generated")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved: %s" % out_path)


if __name__ == "__main__":
    main()
