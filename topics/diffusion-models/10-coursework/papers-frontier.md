# Frontier & Applied Diffusion Papers

Curated applied and recent papers beyond the core DDPM/score-matching foundations. Grouped by theme.

---

## Text-to-image & scaling

**[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)](https://arxiv.org/abs/2205.11487)**
Saharia et al., 2022. Combines a large frozen T5 language model with a cascaded pixel-space diffusion pipeline; demonstrates that text encoder scale matters more than image encoder scale for text fidelity.

**[Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2 / unCLIP)](https://arxiv.org/abs/2204.06125)**
Ramesh et al., 2022. Introduces a two-stage approach: a prior maps CLIP text embeddings to CLIP image embeddings, then a diffusion decoder maps those to pixels; illuminates the CLIP latent space as a useful intermediate for generation and editing.

**[Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)**
Peebles & Xie, 2022. Replaces the U-Net backbone with a Vision Transformer operating on latent patches; shows that standard transformer scaling laws (depth, width, compute) transfer cleanly to diffusion, and that Gflops strongly predict FID.

---

## Faster sampling

**[Consistency Models](https://arxiv.org/abs/2303.01469)**
Song, Dhariwal, Chen, Sutskever, 2023. Trains a model to map any point on a diffusion trajectory directly to the trajectory's origin, enabling single- or few-step generation without a GAN discriminator; achieves near-DDPM quality at a fraction of the NFE budget.

**[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)**
Lipman, Chen, Ben-Hamu, Nickel, Le, 2022. Frames generation as learning an ODE flow between noise and data using simple conditional vector fields; provides straighter trajectories than diffusion, leading to faster and more stable sampling and is the foundation of Stable Diffusion 3 / Flux.

---

## Control & editing

**[Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)](https://arxiv.org/abs/2302.05543)**
Zhang, Rao, Agrawala, 2023. Attaches trainable copies of U-Net encoder blocks to a frozen diffusion backbone; enables spatial conditioning on edges, depth maps, poses, and segmentation masks without retraining the base model.

**[SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)**
Meng et al., 2021. Shows that adding noise to a user-edited image and then running the reverse SDE produces realistic outputs that respect the edit; a simple training-free editing primitive that underlies many later methods.

---

## Beyond images (video / 3D / RL)

**[Video Diffusion Models](https://arxiv.org/abs/2204.03458)**
Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet, 2022. Extends DDPM to spatiotemporal data with a 3D U-Net and joint image-video training; establishes the factored space-time attention blueprint followed by most subsequent video diffusion work.

**[DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)**
Poole, Jain, Barron, Mildenhall, 2022. Introduces Score Distillation Sampling (SDS): optimize a NeRF so that rendered views score highly under a 2D text-conditioned diffusion model, bypassing the need for any 3D training data.

**[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)**
Chi et al., 2023. Applies conditional diffusion to robot action sequences, treating policy learning as denoising over action trajectories conditioned on visual observations; outperforms behavior-cloning and implicit-energy baselines on dexterous manipulation tasks.
