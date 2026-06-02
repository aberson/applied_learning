# Model Distillation — Exercises

Exercises build on the four notebooks in this directory. Work through them in order; each tier assumes the prior tier is solid.

---

## Warm-up

**W1. Temperature sweep — student accuracy vs T**
Notebook: [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Train the same student five times with T in {1, 2, 5, 10, 20}. Plot test accuracy vs T on a single axes.

Hint: T=1 is pure hard-label cross-entropy on soft targets; very high T collapses all logits toward uniform.

Success: Curve shows a clear peak somewhere in [2, 10]; accuracy degrades at extremes. You can explain why in one sentence.

---

**W2. Alpha sweep — soft vs hard loss weight**
Notebook: [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Fix T=4. Train with alpha in {0, 0.25, 0.5, 0.75, 1.0} where alpha weights the soft-target loss and (1-alpha) weights hard-label cross-entropy. Plot test accuracy vs alpha.

Hint: alpha=0 is pure hard-label training; alpha=1 ignores ground-truth labels entirely.

Success: Best alpha is neither 0 nor 1; you can read the optimal off the plot.

---

## Core

**C1. Scratch vs distilled — the headline comparison**
Notebook: [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Train an identical student architecture twice — once from scratch (cross-entropy only) and once with distillation from the pretrained teacher. Report test accuracy for both.

Hint: Use the best (T, alpha) found in W1/W2 for the distilled run.

Success: Distilled student beats scratch-trained by a measurable margin (expect ~0.5–2 pp on MNIST; effect is larger on harder datasets).

---

**C2. Shrink the student until accuracy collapses**
Notebook: [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Define a family of students by halving hidden units at each step (e.g., 256 → 128 → 64 → 32 → 16). Distill the same teacher into each. Plot accuracy vs parameter count.

Hint: Log-scale the x-axis; collapse is usually abrupt rather than gradual.

Success: You can identify an approximate "floor" below which distillation no longer compensates for capacity loss.

---

**C3. Capacity-gap effect — larger teacher, better student?**
Notebook: [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Train two teachers — one small (matches the baseline) and one large (2-4x more parameters). Distill the same student from each. Compare student accuracies.

Hint: A much stronger teacher can produce overconfident, low-entropy soft targets that are less informative for the student.

Success: You observe that the larger teacher does not always yield a better student; write two sentences on why.

---

**C4. Feature / hint distillation on top of logit distillation**
Notebooks: [03-feature-distillation-fitnets](notebooks/03-feature-distillation-fitnets.ipynb), [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Add an intermediate-layer MSE hint loss (FitNets style) to the logit-distillation setup. Compare final accuracy against logit-only distillation at the same student size.

Hint: You need a projection layer if teacher and student hidden dims differ; train the hint phase first, then fine-tune with the full loss.

Success: Feature distillation adds at least a small further gain over logit-only; or you document clearly when it does not and why.

---

**C5. Compression tradeoff plot — params, latency, accuracy**
Notebook: [04-tradeoffs](notebooks/04-tradeoffs.ipynb)

Task: For each student size from C2, measure (a) parameter count, (b) mean inference latency on CPU for a fixed batch, and (c) test accuracy. Produce two plots: accuracy vs params and accuracy vs latency.

Hint: Use `time.perf_counter` around a repeated forward pass (N=100) and take the mean; disable gradients.

Success: Both plots show a Pareto-style curve; you can pick a "sweet spot" and justify it.

---

## Stretch

**S1. Self-distillation / born-again networks**
Notebooks: [01-soft-targets-intuition](notebooks/01-soft-targets-intuition.ipynb), [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Use a trained network as both teacher and student (same architecture). Distill it into a fresh copy of itself. Compare generation-0 vs generation-1 accuracy.

Hint: Born-again networks (Furlanello et al. 2018) show that even same-capacity distillation can improve accuracy through label smoothing and dark knowledge.

Success: Generation-1 matches or beats generation-0 accuracy despite identical architecture; you can chain to generation-2 and see if gains continue.

---

**S2. Ensemble teacher distillation**
Notebook: [02-logit-distillation-mnist](notebooks/02-logit-distillation-mnist.ipynb)

Task: Train three independent teachers (different random seeds). Average their softmax outputs to form an ensemble teacher. Distill the student from the ensemble. Compare against distillation from the single best teacher.

Hint: Average probabilities, not logits, before computing the soft-target loss; the ensemble soft targets are already at T=1 scale.

Success: Ensemble-distilled student outperforms single-teacher-distilled student, demonstrating that soft-label diversity carries useful information.
