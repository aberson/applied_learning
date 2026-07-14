# Exercises — OOD detection

Graded exercise set extending the five notebooks in [`notebooks/`](notebooks/) and
[`examples/ood_min.py`](examples/ood_min.py). All datasets (MNIST, FashionMNIST, KMNIST)
are pre-downloaded at repo-root `data/` — no network calls needed. Each task should run headless
on CPU in seconds; reuse the small subsets/epoch counts already set in the notebooks.

## Warm-up

1. **Swap OOD set: FashionMNIST -> KMNIST** in `02-msp-baseline-mnist.ipynb`, re-measure AUROC/FPR95.
   Notebook: `02-*`. Hint: KMNIST (handwritten kanji) is visually closer to MNIST digits than
   clothing photos — near- vs far-OOD. Check: AUROC should be lower than nb02's FashionMNIST
   result; if it isn't, verify the KMNIST loader uses `download=False` and nb02's exact transform.

2. **Add AUPR** (`sklearn.metrics.average_precision_score`) alongside AUROC. Notebook: `02-*`.
   Hint: reuse the same `labels`/`scores` arrays as the AUROC cell. Check: AUPR should rank methods
   the same way AUROC does on a roughly balanced split; large disagreement signals a label-sign bug.

3. **Add a random-noise OOD set** (`torch.rand` or `torch.randn`, clipped to `[0,1]`, same
   normalization as real images). Notebook: `02-*`. Hint: noise is extreme far-OOD. Check: MSP
   AUROC should land near 0.99+; a mediocre score means the noise batch skipped the notebook's
   normalization pipeline.

## Core

4. **Sweep ODIN's T over `{1, 10, 100, 1000}`** (hold epsilon at nb03's default), plot AUROC vs T.
   Notebook: `03-*`. Hint: vary one grid axis at a time. Check: AUROC should rise then plateau past
   T~100 (matches Liang, Li & Srikant, 2018's ablation) — a flat curve means T isn't reaching the
   softmax before argmax.

5. **Implement FPR@95TPR from scratch**: sort ID scores, take the threshold at the 5th percentile
   (keeps 95% of ID above it), compute the OOD fraction also above it. Notebook: `02-*`. Check:
   should match nb02's reported number to float tolerance; a mismatch is usually an off-by-one on
   the percentile or a flipped ID/OOD label.

6. **Vary k in `{1, 5, 10, 20, 50}`** for the KNN OOD score, plot AUROC vs k. Notebook: `04-*`.
   Hint: k too small overfits to single-point noise; k too large blurs the boundary. Check: expect
   a single-peaked curve (Sun, Ming, Zhu & Li, 2022); note whether nb04's default k sits near the peak.

7. **One shared bar chart**: MSP, ODIN, energy, Mahalanobis, KNN, ensemble entropy on the SAME
   held-out MNIST/FashionMNIST split. Notebooks: `02-*` through `05-*`. Hint: rerun each score
   function on identical test indices, not each notebook's own resample. Check: ordering should
   roughly match the papers (MSP weakest; Mahalanobis/KNN/energy stronger); ~0.5 AUROC for any
   method signals a wiring bug, not a real result.

## Stretch

8. **Near-OOD test**: rotate MNIST 90 degrees, or hold out digit classes 8-9 from training; compare
   Mahalanobis vs energy degradation against nb04/nb03's far-OOD numbers. Hint: rotation keeps the
   domain but shifts the manifold; held-out classes need retraining on digits 0-7 only. Check:
   energy/MSP should drop sharply (toward 0.5-0.7 AUROC) while Mahalanobis holds up better — the
   feature-space geometry argument from Lee, Lee, Lee & Shin, 2018.

9. **Outlier Exposure mini-experiment**: add KMNIST as an auxiliary uniform-target cross-entropy
   term during nb02's training (weight ~0.5x main loss), re-measure MSP AUROC on FashionMNIST.
   Notebook: `02-*`. Check: AUROC should rise over the un-exposed baseline (Hendrycks, Mazeika &
   Dietterich, 2019); confirm ID test accuracy didn't regress.

10. **Threshold-setting without OOD labels**: set the cutoff at the 95th percentile of the ID-only
    score distribution (MNIST test split only), then report the realized FPR on OOD. Notebook:
    `02-*`. Check: realized FPR should be in the same ballpark as nb02's FPR@95TPR (same threshold,
    different derivation) — this is the deployable protocol, since production has no OOD labels
    at threshold-setting time.

11. **Relative Mahalanobis** (Ren, Fort, Liu, Roy, Padhy & Lakshminarayanan, 2021): fit one extra
    Gaussian on all pooled training features (ignoring class), then score
    `RMD(x) = Mahalanobis_class(x) - Mahalanobis_background(x)`. Re-measure on exercise 8's near-OOD
    set. Notebook: `04-*`. Check: AUROC should improve specifically on near-OOD versus plain
    Mahalanobis, with little change on far-OOD (FashionMNIST) — RMD targets exactly that failure mode.

See [`../10-coursework/papers-seminal.md`](../10-coursework/papers-seminal.md) and
[`papers-frontier.md`](../10-coursework/papers-frontier.md) for full citations.
