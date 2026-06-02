# Diffusion Models — Open Questions & Mentoring Prompts

Use these after working through the [essentials](../10-coursework/essentials.md) and hands-on notebooks. Self-check questions are things you should be able to answer cold; deeper threads are starting points for a conversation.

---

## Can you explain it? (self-check)

1. **Noise prediction vs. x_0 prediction.** The network is usually trained to predict the noise added at step t, not the clean image directly. Why does this choice tend to produce sharper samples? What changes if you parameterize it as x_0 prediction instead?

2. **Role of alpha_bar_t.** What does the cumulative noise schedule alpha_bar_t control? Sketch roughly what x_t looks like at alpha_bar_t near 1 vs. near 0.

3. **Why the U-Net with time conditioning?** What architectural job does the U-Net's skip-connection structure do for diffusion, and why must the network also accept t as an input?

4. **DDIM speedup.** DDPM requires ~1000 steps; DDIM can use ~50 and still look sharp. What is the key insight that allows this, and what is the theoretical tradeoff being made?

5. **Classifier-free guidance scale.** During inference you interpolate between the conditional and unconditional score estimate with a scale w. What does increasing w do to sample quality, and what does it cost?

6. **Variance schedule design.** Linear vs. cosine noise schedules produce different training dynamics. What failure mode motivated the cosine schedule, and how does it differ qualitatively from linear?

7. **Latent vs. pixel diffusion.** Stable Diffusion runs the diffusion process in a compressed latent space. What is the VAE doing here, and what does operating in latent space buy you in practice?

8. **Score function connection.** Diffusion models are sometimes described as learning a score function (gradient of log p(x)). In one or two sentences, what is the score, and how does the noise-prediction network approximate it?

---

## Deeper threads

1. **SDE vs. ODE view.** Both framings are equivalent at the distribution level. When is the SDE framing more useful (e.g., for theory, for samplers)? When would you prefer the probability-flow ODE framing?

2. **Consistency and flow-matching models.** These claim competitive quality in one or a handful of steps. What is the core argument for why they can do this? Are they fundamentally different from DDIM, or refinements of the same idea?

3. **Failure modes and diagnostics.** What does a diffusion model generate when the guidance scale is too high? When the number of inference steps is too low? When the model is undertrained? How would you distinguish these in practice?

4. **Text-image alignment vs. photorealism.** Guidance increases prompt adherence but can reduce diversity and introduce artifacts. Is there a principled way to choose the guidance scale for a given use case, or is it always empirical?

5. **Discrete vs. continuous diffusion.** Standard diffusion is defined over continuous pixel/latent values. How do you extend the framework to discrete tokens (text, code)? What breaks, and what are the proposed fixes?

6. **When not to use diffusion.** Diffusion models are slow at inference and require significant compute. What application constraints (latency, compute budget, data volume) would push you toward a GAN, VAE, or autoregressive model instead?

---

## Things to try next

1. **Custom scheduler comparison.** Swap the noise scheduler in your notebook (DDPM → DDIM → DPM-Solver) on the same prompt and count the steps needed to reach equivalent quality. Log the wall-clock time per run.

2. **LoRA fine-tuning on a small domain.** Take a pretrained Stable Diffusion checkpoint and fine-tune a LoRA adapter on 20–50 domain images. Observe how quickly the style transfers and where the model fails.

3. **Classifier-free guidance ablation.** Generate a grid of the same prompt with guidance scale from 1 to 20. Document where sharpness peaks, where artifacts appear, and how diversity changes across the grid.

4. **Inpainting and editing pipelines.** Extend the basic text-to-image notebook to do inpainting (mask a region and re-diffuse). Then try prompt-to-prompt or instruct-pix2pix style edits and note what additional conditioning mechanism each requires.
