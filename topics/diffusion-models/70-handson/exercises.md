# Diffusion Models — Hands-On Exercises

Exercises build on the four notebooks in `./notebooks/`. Work through them in order; each tier assumes the prior one is solid.

---

## Warm-up

**W1. Visualize the full forward process**
In [01-forward-noising-intuition](notebooks/01-forward-noising-intuition.ipynb), plot the noisy sample at every 100th timestep (t = 0, 100, 200, … T) as a grid of images. Confirm visually that the signal is gone by t = T.
*Hint:* call `q_sample(x0, t)` for each t and use `matplotlib.gridspec`.
*Success:* grid shows a smooth transition from clean image to isotropic Gaussian noise.

**W2. Swap linear for cosine noise schedule**
In [02-ddpm-from-scratch-2d](notebooks/02-ddpm-from-scratch-2d.ipynb), replace the linear beta schedule with the cosine schedule from Nichol & Dhariwal (2021). Recompute `alphas_cumprod` and plot both schedules on the same axes.
*Hint:* cosine schedule: `alphas_cumprod[t] = cos((t/T + s)/(1+s) * pi/2)^2` normalized by `alphas_cumprod[0]`.
*Success:* cosine curve stays near 1.0 longer, drops sharply near T; the two curves visibly differ.

**W3. Reduce T and watch quality degrade**
In [02-ddpm-from-scratch-2d](notebooks/02-ddpm-from-scratch-2d.ipynb), halve T (e.g., 500 → 250) without retraining. Sample and overlay the generated points on the training set.
*Hint:* only the sampling loop changes; keep the trained model fixed.
*Success:* samples spread out or collapse relative to the T=500 baseline; you can articulate why.

---

## Core

**C1. Visualize the reverse trajectory of a single sample**
In [02-ddpm-from-scratch-2d](notebooks/02-ddpm-from-scratch-2d.ipynb) or [03-ddpm-mnist](notebooks/03-ddpm-mnist.ipynb), save the denoised estimate `x_0_hat` at every 50th reverse step for one sample and animate or grid-plot the trajectory.
*Hint:* store intermediate states inside the sampling loop before the reparametrization step.
*Success:* trajectory shows a noisy blob converging to a recognizable sample; earlier frames are visibly noisier.

**C2. Compare DDIM vs ancestral (DDPM) at different step counts**
In [04-ddim-and-guidance](notebooks/04-ddim-and-guidance.ipynb), run both samplers at step counts {10, 25, 50, 200} using the same trained model. Plot or tabulate visual quality (or nearest-neighbor distance to training set) vs. step count for each sampler.
*Hint:* DDIM requires a subsequence `{t_1, …, t_S}` ⊂ {0, …, T}; pick it uniformly spaced.
*Success:* DDIM degrades more gracefully than DDPM at low step counts; results match the paper's qualitative claim.

**C3. Train on a different 2D distribution**
In [02-ddpm-from-scratch-2d](notebooks/02-ddpm-from-scratch-2d.ipynb), swap the training dataset for `sklearn.datasets.make_swiss_roll` (projected to 2D) or `make_moons`. Retrain and sample.
*Hint:* only the dataset generation cell changes; all training/sampling code is reused.
*Success:* generated samples match the shape of the new distribution; density is qualitatively correct.

**C4. Add classifier-free guidance and sweep the scale**
In [04-ddim-and-guidance](notebooks/04-ddim-and-guidance.ipynb), implement classifier-free guidance: train a conditional model with 10% dropout on the conditioning signal, then at sampling time compute `eps_guided = eps_uncond + w * (eps_cond - eps_uncond)` and sweep `w` over {0, 1, 3, 7}.
*Hint:* represent "no condition" with a null token (e.g., class index = num_classes).
*Success:* higher `w` produces sharper but less diverse samples; `w=0` matches unconditional baseline.

---

## Stretch

**S1. Condition the MNIST model on digit label**
In [03-ddpm-mnist](notebooks/03-ddpm-mnist.ipynb), add a learned class embedding (10 classes) summed into the time embedding and retrain. At inference, specify a target digit.
*Hint:* `nn.Embedding(10, time_emb_dim)` added to the UNet's time projection path.
*Success:* sampling with label=3 produces recognizable 3s; a grid of one sample per class shows clear class separation.

**S2. Measure sample quality with nearest-neighbor distance**
After any sampling run (2D or MNIST), compute the mean nearest-neighbor distance from each generated sample to the training set in pixel/feature space.
*Hint:* for MNIST, flatten images to vectors; use `sklearn.neighbors.NearestNeighbors` with `n_neighbors=1`.
*Success:* reported metric decreases as model training progresses or as guidance scale increases (up to a point), giving a crude FID-free quality signal.
