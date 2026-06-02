# Related Topics — Diffusion Models Adjacency Map

Neighboring concepts a diffusion learner should know, with their relationship to diffusion (prerequisite, alternative, generalization, or application).

---

**Variational Autoencoders (VAEs)**
Learn a latent space by maximizing a lower bound on data log-likelihood (ELBO), using an encoder-decoder pair with a KL-divergence regularizer. Prerequisite for latent diffusion: the autoencoder in Stable Diffusion is a VAE (or a KL-regularized variant). Understanding the reparameterization trick and ELBO grounds the noisy-latent intuition in diffusion.

**Generative Adversarial Networks (GANs)**
Train a generator against a discriminator in a minimax game; historically the dominant image-synthesis method. Alternative: diffusion models produce higher-diversity samples and are more stable to train, but are slower at inference. Score-based views of diffusion borrow the "learn to distinguish" intuition but replace adversarial training with denoising regression.

**Normalizing Flows**
Exact-likelihood generative models that learn an invertible mapping from data to a simple prior. Alternative and generalization: flow matching (below) relaxes the invertibility constraint to straight-line paths, inheriting flow vocabulary while escaping the Jacobian cost. Diffusion and flows both sit inside the broader framework of continuous-time generative models.

**Energy-Based Models (EBMs)**
Define an unnormalized probability via a scalar energy function; sampling requires MCMC (e.g., Langevin dynamics). Score matching (below) was originally developed to train EBMs without computing the partition function. Diffusion can be read as an EBM whose score function is parameterized by a U-Net trained via denoising score matching.

**Score Matching and Langevin Dynamics**
Score matching (Hyvarinen 2005) trains a model to match the gradient of the log-density (the "score") without the normalization constant. Langevin dynamics uses iterated noisy gradient ascent on that score to sample. Direct prerequisite: Song & Ermon (2019) showed that a hierarchy of noise scales makes score matching practical at data scale, which is the conceptual core of DDPM's forward/reverse process. See [essentials](essentials.md) for the denoising score matching objective.

**Stochastic Differential Equations / Probability-Flow ODE**
Song et al. (2021) unified diffusion as a continuous-time SDE (forward: data → noise; reverse: learned drift). The probability-flow ODE is the deterministic counterpart: same marginals, no stochasticity, enables exact likelihood computation and faster deterministic sampling. Generalization: SDE/ODE framing subsumes DDPM, SMLD, and most subsequent samplers under one formalism.

**Flow Matching / Rectified Flow**
Flow matching (Lipman et al. 2022) trains a velocity field to transport noise to data along simple paths, bypassing the score-matching objective entirely. Rectified flow (Liu et al. 2022) further straightens trajectories by iterative reflow, enabling one- or few-step generation. Successor/alternative: avoids the variance-explosion forward process and trains faster; now the backbone of many production text-to-image systems.

**Consistency Models**
Distill a diffusion trajectory into a model that maps any noisy point directly to the clean endpoint in one step (Song et al. 2023). Application: inference-time speedup without retraining the base model from scratch. Consistency training is an alternative that skips distillation. Trade-off vs. flow matching: both target fast sampling but take different theoretical routes.

**Autoregressive Generative Models**
Model p(x) as a product of conditionals over tokens (PixelCNN, GPT-style image models, AudioLM). Alternative: diffusion denoises all positions jointly and scales better to continuous data without discretization. Current research (e.g., MAR, discrete diffusion) blends the two — masked autoregressive steps treated as a diffusion process over token space.

**Latent Diffusion / Autoencoders**
Run the diffusion process in a compressed latent space encoded by a pre-trained autoencoder (Rombach et al. 2022). Application and substrate: reduces quadratic pixel-space compute; enables training on high-resolution images. The autoencoder and the diffusion U-Net are decoupled, so each can be swapped independently. See [seminal papers](papers-seminal.md) for the LDM paper.

**CLIP and Guidance**
CLIP (Radford et al. 2021) jointly trains image and text encoders on web-scale pairs. In diffusion: classifier guidance (Dhariwal & Nichol 2021) used a separately trained classifier gradient to steer samples; classifier-free guidance (Ho & Salimans 2021) inlines conditioning by jointly training a conditional and unconditional model and interpolating at inference. Prerequisite for text-to-image: conditioning the denoiser on CLIP or T5 text embeddings is how prompts drive generation.

**Classifier-Free Guidance (CFG)**
At inference, score = unconditional score + w × (conditional − unconditional score); higher w increases text alignment at the cost of diversity. Practical prerequisite: virtually every production diffusion model uses CFG. Understanding it is necessary to interpret generation quality vs. diversity trade-offs and newer distillation targets. See [essentials](essentials.md).

**Samplers: DDIM, DPM-Solver, Ancestral**
Ancestral sampling follows the exact DDPM reverse chain (1000 steps). DDIM (Song et al. 2020) reformulates as a non-Markovian process and enables deterministic 10-50 step generation. DPM-Solver (Lu et al. 2022) treats the reverse ODE analytically for order-2/3 accuracy with fewer NFE. Application: sampler choice controls the quality/speed frontier at inference without retraining.

**Diffusion for Audio**
WaveGrad, DiffWave, and AudioLDM apply diffusion in raw waveform or mel-spectrogram space. Application domain: the forward/reverse process is unchanged; the architecture adapts (1-D conv or spectrogram U-Net). Conditional generation on text or reference audio follows the same CFG machinery.

**Diffusion for Video**
Extend the U-Net with temporal attention or 3-D convolutions; add temporal consistency conditioning. Application domain: generation quality degrades with sequence length due to compounding errors; flow matching and consistency models are active targets for reducing inference cost.

**Diffusion for 3-D and Molecules**
Point-cloud diffusion (e.g., LION, DiffSDF) operates on SE(3)-equivariant representations. Molecule generation (DiffSBDD, EquiDiff) adds torsion-angle or atom-type channels. Application domain: equivariance constraints replace standard U-Net backbones; the score-matching objective is unchanged.

**Diffusion Policy / RL**
Diffuser (Janner et al. 2022) and Diffusion Policy treat trajectory planning or action generation as a denoising problem over state-action sequences. Application domain: diffusion's multimodal distribution over actions captures behavioral diversity better than deterministic policies. The reverse process is conditioned on goal or observation context instead of a text prompt.
