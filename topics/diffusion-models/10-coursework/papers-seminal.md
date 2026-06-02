# Seminal Diffusion Models — Reading List

**Suggested reading order:** DDPM → DDIM → Score SDE → Classifier Guidance → Classifier-Free Guidance → Latent Diffusion → EDM; then loop back to Sohl-Dickstein 2015 and NCSN for historical grounding; finish with iDDPM for implementation details.

---

## Start here

### [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
Ho, Jain, Abbeel (2020)

Establishes the modern DDPM formulation: a fixed forward noising process and a learned reverse denoising network trained with a simplified loss (L_simple). The paper every subsequent diffusion work builds on; start here before reading anything else.

---

## Foundations

### [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)

The original proposal to learn a generative model by reversing a diffusion process borrowed from non-equilibrium statistical physics. Read after DDPM to understand where the core idea came from; the modern treatment is cleaner, but this is the conceptual origin.

### [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)
Song & Ermon (2019)

Introduces Noise Conditional Score Networks (NCSN) and score matching as a route to generation via Langevin dynamics. Establishes the score-function perspective that later unifies with DDPM under the SDE framework.

---

## Core improvements and sampling

### [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
Song, Meng, Ermon (2020)

Reformulates DDPM sampling as a deterministic (non-Markovian) process, enabling 10–50× speedups with no retraining. Understanding DDIM is essential before studying any later fast-sampling or consistency-distillation work.

### [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
Nichol & Dhariwal (2021)

Adds a cosine noise schedule and learned variance to DDPM, improving log-likelihood and sample quality on standard benchmarks. A concise read that motivates several engineering choices seen in production implementations.

---

## Guidance

### [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
Dhariwal & Nichol (2021)

Introduces classifier guidance — using gradients of a separately trained classifier to steer the reverse diffusion toward a target class, demonstrating that diffusion models can surpass GANs on FID. Establishes the conditioning paradigm extended by subsequent guidance work.

### [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
Ho & Salimans (2022)

Removes the need for a separate classifier by jointly training a conditional and unconditional model, then interpolating at inference. The technique used in virtually every text-to-image system; read immediately after Dhariwal & Nichol 2021.

---

## Unification and theory

### [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021)

Frames both DDPM and NCSN as special cases of a continuous-time SDE framework, providing a unified theoretical lens and new deterministic probability-flow ODE samplers. The deepest theoretical paper in this list; read after DDPM + NCSN + DDIM.

---

## Architecture and scaling

### [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
Rombach, Blattmann, Lorenz, Esser, Ommer (2022)

Moves diffusion into a compressed latent space learned by a VQ-regularized autoencoder, dramatically reducing compute while preserving quality. The architecture behind Stable Diffusion; required reading for understanding modern text-to-image pipelines.

---

## Design-space analysis

### [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)
Karras, Aittala, Aila, Laine (2022)

Systematically separates and reparameterizes the noise schedule, network preconditioning, and sampler, showing that careful design choices yield large quality gains independent of architecture changes. Essential for practitioners tuning or ablating diffusion pipelines.
