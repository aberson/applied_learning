# Diffusion Models — Core Concepts Reference

Terse intuition-first reference. Grep-friendly. Math uses inline notation; LaTeX rendering not required to be useful.

---

## Core Idea

A diffusion model has two processes:

- **Forward process q**: destroy data by adding Gaussian noise over T steps until the signal is indistinguishable from pure noise.
- **Reverse process p_theta**: learn to undo that destruction step by step, so that starting from noise and running the reverse process generates new data.

Training teaches a neural network to denoise. Inference runs denoising from scratch.

---

## Forward Process q(x_t | x_0)

A Markov chain of T steps, each adding a small amount of Gaussian noise controlled by a scalar beta_t in (0,1):

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

**Closed form (key identity).** Define alpha_t = 1 - beta_t and alpha_bar_t = prod_{s=1}^{t} alpha_s. Then:

```
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

This lets you jump directly to any noise level without simulating intermediate steps.

**Reparameterization** (used in every training step):

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
```

As t -> T, alpha_bar_t -> 0, so x_t -> pure noise. As t -> 0, x_t -> x_0.

---

## Training Objective

The true ELBO objective simplifies (Ho et al. 2020, DDPM) to a **noise-prediction loss**:

```
L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t) ||^2 ]
```

- Sample a data point x_0, a timestep t ~ Uniform(1..T), and noise eps ~ N(0,I).
- Corrupt: compute x_t via the reparameterization above.
- Train eps_theta (typically a U-Net conditioned on t) to predict eps from x_t and t.

**Why predict noise?** Because x_t is a known linear combination of x_0 and eps, predicting eps is equivalent to predicting x_0, but empirically more stable. Predicting eps also directly connects to the score function (see Reverse Sampling below).

---

## Reverse Sampling (Ancestral Sampling)

The reverse step (DDPM):

```
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
```

where z ~ N(0,I) and sigma_t is a schedule-dependent noise level (often sqrt(beta_t) or a posterior variance estimate).

**Score connection.** The score of the data distribution is grad_{x_t} log p(x_t). The noise prediction is related by:

```
eps_theta(x_t, t) ~ -sqrt(1 - alpha_bar_t) * score(x_t, t)
```

So a diffusion model is simultaneously learning the score function, connecting to score-based generative models.

---

## Noise Schedules

The beta schedule controls how quickly signal is destroyed.

- **Linear** (Ho et al. 2020): beta_t rises linearly from ~1e-4 to ~0.02. Works but destroys signal quickly at low t, wasting early timesteps.
- **Cosine** (Nichol & Dhariwal 2021): alpha_bar_t follows a cosine curve; smoother signal decay, better sample quality especially at low resolution.

T is typically 1000. This makes naive sampling slow: 1000 forward passes through the network per sample.

---

## Why Sampling Is Slow, and the Fix

Each denoising step requires one network forward pass. T=1000 steps per image is expensive.

**DDIM** (Song et al.) reformulates the reverse process as a non-Markovian ODE-style update, deterministic by default, that can skip steps. With 50 DDIM steps you get comparable quality to 1000 DDPM steps.

---

## Map of Variants You Will Meet

- **DDPM** (Ho et al. 2020): the baseline; stochastic ancestral sampling, T=1000, noise prediction loss.
- **DDIM**: deterministic sampler derived from DDPM; same trained model, much faster inference by skipping timesteps.
- **Score/SDE view** (Song et al. 2021): unifies diffusion models as stochastic differential equations; forward SDE adds noise continuously, reverse SDE generates samples; DDPM and SMLD are special cases.
- **Classifier-free guidance (CFG)**: condition the model on a label/text; at inference, interpolate between conditional and unconditional predictions to trade diversity for fidelity. No separate classifier needed.
- **Latent diffusion (LDM / Stable Diffusion)**: run the diffusion process in the latent space of a pretrained VAE, not in pixel space. Dramatically cheaper; enables high-resolution generation.

---

## Math Prerequisites

- **Gaussians**: products, conditioning, KL divergence between Gaussians in closed form.
- **KL divergence**: the ELBO derivation involves KL(q || p) terms; you need to know why minimizing KL is minimizing negative log-likelihood.
- **Conditional expectation**: the tower property is used throughout the ELBO derivation.
- **Reparameterization trick**: sampling from N(mu, sigma^2) as mu + sigma * eps; essential for the x_t closed form and for VAE connections.
- **SDEs/ODEs** (optional for DDPM; needed for score/SDE view): Ito calculus, Langevin dynamics, the link between the reverse SDE and the score function.
