"""Minimal, self-contained logit-distillation demo (torch, CPU).

What it does
------------
Trains a TEACHER MLP on the sklearn `load_digits` dataset (8x8 handwritten
digit images, ~1797 samples, ships with scikit-learn -- NO download needed),
then trains a tiny STUDENT MLP two ways:

  1. from scratch  -- standard cross-entropy on the hard labels.
  2. distilled     -- knowledge distillation from the teacher's logits using
                      loss = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
                           + (1 - alpha) * CE(z_s, y)
                      with temperature T=4 and alpha=0.7.

It prints a small ASCII results table (teacher / student-scratch /
student-distilled test accuracy + parameter counts) and saves a 3-bar
accuracy comparison figure to `acc_comparison.png` next to this script.

Note on the expected outcome
----------------------------
`load_digits` is a small, easy dataset: a ~14x smaller student already
matches the teacher closely from hard labels alone, so the scratch vs.
distilled gap is typically near zero (often within +/- 0.01) and its sign
varies run-to-run. The point of this demo is the mechanics of the
distillation loss, not a guaranteed accuracy win -- distillation's edge grows
on harder datasets and more under-capacity students. The honest, reproducible
take-away here is "a tiny student recovers near-teacher accuracy", with
distillation roughly tying the from-scratch baseline.

How to run
----------
From this examples directory:

    uv run python distill_min.py

(or from the project root:
    uv run --project c:/Users/abero/dev/applied_learning python \
        topics/model-distillation/70-handson/examples/distill_min.py)

Runs headless on CPU in well under ~30s. Writes `acc_comparison.png` next to
this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cpu")
TEMPERATURE = 4.0
ALPHA = 0.7
N_EPOCHS = 120


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TeacherMLP(nn.Module):
    """A couple of hidden layers -- the larger, more capable model."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentMLP(nn.Module):
    """A tiny single-hidden-layer model -- the compact model to deploy."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        return (preds == y).float().mean().item()


def train_hard(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = N_EPOCHS,
    lr: float = 1e-2,
) -> None:
    """Standard supervised training on hard labels (CE loss)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()


def train_distilled(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    temperature: float = TEMPERATURE,
    alpha: float = ALPHA,
    epochs: int = N_EPOCHS,
    lr: float = 1e-2,
) -> None:
    """Train the student with KL-on-soft-targets + hard-label CE.

    loss = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
         + (1 - alpha) * CE(z_s, y)
    """
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(x)
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(epochs):
        student.train()
        opt.zero_grad()
        z_s = student(x)
        # KLDivLoss expects log-probabilities as input, probabilities as target.
        log_soft_student = F.log_softmax(z_s / temperature, dim=1)
        kl = F.kl_div(log_soft_student, soft_targets, reduction="batchmean")
        # T^2 keeps gradient magnitudes comparable across temperatures.
        distill_loss = alpha * (temperature**2) * kl
        hard_loss = (1.0 - alpha) * F.cross_entropy(z_s, y)
        loss = distill_loss + hard_loss
        loss.backward()
        opt.step()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # --- Data -------------------------------------------------------------- #
    digits = load_digits()
    x_all = digits.data.astype("float32")  # (n_samples, 64)
    y_all = digits.target.astype("int64")
    n_classes = int(y_all.max()) + 1

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_all, y_all, test_size=0.25, random_state=0, stratify=y_all
    )

    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr).astype("float32")
    x_te = scaler.transform(x_te).astype("float32")

    x_tr_t = torch.from_numpy(x_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    x_te_t = torch.from_numpy(x_te).to(DEVICE)
    y_te_t = torch.from_numpy(y_te).to(DEVICE)

    in_dim = x_tr_t.shape[1]

    # --- (1) Teacher ------------------------------------------------------- #
    teacher = TeacherMLP(in_dim, n_classes).to(DEVICE)
    train_hard(teacher, x_tr_t, y_tr_t)
    teacher_acc = accuracy(teacher, x_te_t, y_te_t)

    # --- (2a) Student from scratch (hard labels) --------------------------- #
    student_scratch = StudentMLP(in_dim, n_classes).to(DEVICE)
    train_hard(student_scratch, x_tr_t, y_tr_t)
    scratch_acc = accuracy(student_scratch, x_te_t, y_te_t)

    # --- (2b) Student distilled from teacher ------------------------------- #
    student_distilled = StudentMLP(in_dim, n_classes).to(DEVICE)
    train_distilled(student_distilled, teacher, x_tr_t, y_tr_t)
    distilled_acc = accuracy(student_distilled, x_te_t, y_te_t)

    # --- (3) Results table (ASCII only) ------------------------------------ #
    rows = [
        ("teacher", teacher_acc, count_params(teacher)),
        ("student-scratch", scratch_acc, count_params(student_scratch)),
        ("student-distilled", distilled_acc, count_params(student_distilled)),
    ]
    print("")
    print("Logit distillation on sklearn load_digits (T=%g, alpha=%g)" % (TEMPERATURE, ALPHA))
    print("-" * 52)
    name_w = max(len(name) for name, _, _ in rows)
    for name, acc, params in rows:
        print("%-*s  acc=%.3f  params=%d" % (name_w, name, acc, params))
    print("-" * 52)
    delta = distilled_acc - scratch_acc
    print("distillation delta (distilled - scratch) = %+.3f" % delta)
    print("(load_digits is easy: scratch vs distilled is typically a near-tie;")
    print(" distillation's edge grows on harder data / smaller students.)")
    print("")

    # --- (4) 3-bar accuracy comparison figure ------------------------------ #
    out_path = Path(__file__).resolve().parent / "acc_comparison.png"
    labels = ["teacher", "student\nscratch", "student\ndistilled"]
    accs = [teacher_acc, scratch_acc, distilled_acc]
    colors = ["#4C72B0", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Distillation: teacher vs student (scratch vs distilled)")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            "%.3f" % acc,
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Saved figure to %s" % out_path)


if __name__ == "__main__":
    main()
