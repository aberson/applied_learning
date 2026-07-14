"""
ood_min.py -- minimal, self-contained distillation of out-of-distribution (OOD) detection.

WHAT THIS DOES
    1. Builds a SYNTHETIC in-distribution (ID) problem: 3 tight Gaussian class
       clusters in 2D. No dataset download, no network.
    2. Builds a SYNTHETIC out-of-distribution (OOD) set: points far outside the
       ID support (a distant Gaussian blob + uniform background noise).
    3. Trains a tiny torch MLP classifier on the ID classes (CPU, a few hundred
       steps, runs in seconds).
    4. Scores held-out ID vs OOD points with two classic OOD scores:
         - MSP    = max softmax probability      (lower  => more OOD)
         - Energy = -logsumexp(logits)           (higher => more OOD)
    5. Reports AUROC for each score, oriented so OOD is the positive/detected
       class (AUROC > 0.5 means the score usefully separates OOD from ID).
    6. Saves one figure (ood_scores.png, next to this script): overlaid
       histograms of the energy score for ID vs OOD, titled with both AUROCs.

HOW TO RUN
    From the applied_learning repo root:
        uv run python topics/out-of-distribution-detection/70-handson/examples/ood_min.py

    Or directly if this project's venv is already active:
        uv run python ood_min.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Synthetic data: ID = 3 tight Gaussian clusters in 2D
# ---------------------------------------------------------------------------
def make_id_data(n_per_class=200, std=0.35):
    """3 well-separated, tight Gaussian class clusters in 2D."""
    centers = np.array([
        [-3.0, -3.0],
        [3.0, -3.0],
        [0.0, 3.0],
    ])
    xs, ys = [], []
    for label, c in enumerate(centers):
        pts = c + std * np.random.randn(n_per_class, 2)
        xs.append(pts)
        ys.append(np.full(n_per_class, label))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return x, y


def make_ood_data(n_far=300, n_uniform=300):
    """OOD = a distant cluster well outside ID support + uniform background noise."""
    far_cluster = np.array([9.0, 9.0]) + 0.5 * np.random.randn(n_far, 2)
    uniform_noise = np.random.uniform(low=-12.0, high=12.0, size=(n_uniform, 2))
    x = np.concatenate([far_cluster, uniform_noise], axis=0).astype(np.float32)
    return x


# Build train / test ID split, plus the OOD eval set.
x_id_train, y_id_train = make_id_data(n_per_class=200)
x_id_test, y_id_test = make_id_data(n_per_class=100)  # fresh draw = held-out ID
x_ood = make_ood_data(n_far=150, n_uniform=150)

x_id_train_t = torch.from_numpy(x_id_train)
y_id_train_t = torch.from_numpy(y_id_train)
x_id_test_t = torch.from_numpy(x_id_test)
x_ood_t = torch.from_numpy(x_ood)


# ---------------------------------------------------------------------------
# 2. Tiny MLP classifier
# ---------------------------------------------------------------------------
class TinyMLP(nn.Module):
    # NOTE on activation choice: plain ReLU MLPs are known to grow MORE
    # confident (not less) far from the training data (Hein et al. 2019,
    # "Why ReLU Networks Yield High-Confidence Predictions Far Away From the
    # Training Data") -- ReLU regions extend to infinity, so logit
    # differences keep growing with distance. In this synthetic setup that
    # INVERTS both OOD scores (measured AUROC < 0.5 with ReLU here). Tanh
    # saturates, which bounds the logits' growth so confidence properly
    # decays away from the ID clusters -- that's why it's used below.
    def __init__(self, in_dim=2, hidden=32, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


model = TinyMLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

N_STEPS = 300
model.train()
for step in range(N_STEPS):
    optimizer.zero_grad()
    logits = model(x_id_train_t)
    loss = loss_fn(logits, y_id_train_t)
    loss.backward()
    optimizer.step()

model.eval()


# ---------------------------------------------------------------------------
# 3. OOD scores: MSP and Energy
# ---------------------------------------------------------------------------
def msp_score(logits):
    """Max softmax probability. Lower = more OOD."""
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values.detach().numpy()


def energy_score(logits, T=1.0):
    """Energy = -T * logsumexp(logits / T). Higher = more OOD."""
    energy = -T * torch.logsumexp(logits / T, dim=1)
    return energy.detach().numpy()


with torch.no_grad():
    logits_id_test = model(x_id_test_t)
    logits_ood = model(x_ood_t)

    # Accuracy on held-out ID test points.
    preds_id_test = logits_id_test.argmax(dim=1).numpy()
    id_accuracy = float((preds_id_test == y_id_test).mean())

    msp_id = msp_score(logits_id_test)
    msp_ood = msp_score(logits_ood)

    energy_id = energy_score(logits_id_test)
    energy_ood = energy_score(logits_ood)


# ---------------------------------------------------------------------------
# 4. AUROC (OOD = positive/detected class, oriented so AUROC > 0.5)
# ---------------------------------------------------------------------------
# Labels: 0 = ID, 1 = OOD.
labels = np.concatenate([np.zeros_like(msp_id), np.ones_like(msp_ood)])

# MSP is lower for OOD, so use -MSP as the "OOD-ness" score (higher = more OOD).
msp_ood_ness = np.concatenate([-msp_id, -msp_ood])
auroc_msp = roc_auc_score(labels, msp_ood_ness)

# Energy is already higher for OOD, use it directly as the "OOD-ness" score.
energy_ood_ness = np.concatenate([energy_id, energy_ood])
auroc_energy = roc_auc_score(labels, energy_ood_ness)


# ---------------------------------------------------------------------------
# 5. Plot: overlaid histograms of the energy score, ID vs OOD
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(
    min(energy_id.min(), energy_ood.min()),
    max(energy_id.max(), energy_ood.max()),
    40,
)
ax.hist(energy_id, bins=bins, alpha=0.6, label="ID (held-out)", color="#4C72B0", density=True)
ax.hist(energy_ood, bins=bins, alpha=0.6, label="OOD (far cluster + uniform noise)", color="#C44E52", density=True)
ax.set_xlabel("Energy score = -logsumexp(logits)  (higher = more OOD)")
ax.set_ylabel("density")
ax.set_title(
    f"OOD detection via Energy score\n"
    f"AUROC(Energy) = {auroc_energy:.3f}   AUROC(MSP) = {auroc_msp:.3f}"
)
ax.legend()
fig.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ood_scores.png")
fig.savefig(out_path, dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 6. ASCII-only summary
# ---------------------------------------------------------------------------
print("=== OOD detection minimal example ===")
print(f"ID train points: {len(x_id_train)} | ID test points: {len(x_id_test)} | OOD points: {len(x_ood)}")
print(f"ID test accuracy: {id_accuracy:.3f}")
print(f"AUROC (MSP, OOD=positive):    {auroc_msp:.3f}")
print(f"AUROC (Energy, OOD=positive): {auroc_energy:.3f}")
print(f"Saved figure to: {out_path}")
print("Done.")
