# Diffusion Models — Courses & Resources

## Blogs & Tutorials

**"What are Diffusion Models?"** — Lilian Weng (lilianweng.github.io)
Rigorous walkthrough of DDPM, score matching, and SDE formulations with clear math. Use for building intuition and seeing the major formulations side-by-side before diving into papers.

**"Generative Modeling by Estimating Gradients of the Data Distribution"** — Yang Song (yang-song.net)
The author of NCSN/score-based diffusion explains the score-matching perspective from first principles. Essential reading for understanding the score-function view that unifies many methods.

**"The Annotated Diffusion Model"** — Hugging Face (huggingface.co/blog/annotated-diffusion)
Line-by-line PyTorch implementation of DDPM modeled after "The Annotated Transformer." Use when you want to go from math to working code in one sitting; strong on the training loop and noise schedule details.

**Understanding Diffusion Models: A Unified Perspective** — Calvin Luo (arXiv 2208.11970)
Tutorial-length paper (~50 pages) that derives DDPM, score matching, and flow-based views from a single probabilistic lens. Use when you want the math to click all at once rather than reading three separate papers.

---

## Code / Hands-On

**Hugging Face Diffusers library** — Hugging Face (huggingface.co/docs/diffusers)
The standard production library for diffusion pipelines (DDPM, DDIM, Stable Diffusion, ControlNet, etc.). Docs include conceptual guides, pipeline API reference, and worked examples. Use as your primary coding substrate.

**Hugging Face Diffusion Models Class** — Hugging Face (github.com/huggingface/diffusion-models-class)
Free four-unit course built around Diffusers notebooks: trains a DDPM from scratch, fine-tunes Stable Diffusion, covers guidance and ControlNet. The most direct path from zero to running experiments.

---

## Video Lectures

**Practical Deep Learning for Coders, Part 2** — fast.ai / Jeremy Howard
Part 2 covers the Stable Diffusion stack from scratch, including CLIP, VAE, U-Net, and the diffusion loop. Taught in notebook-first style; strong on intuition and "how do I actually run this" before theory. Check fast.ai for the current Part 2 course link.

**Stanford CS236 / CS330 guest lectures on score-based models** — various instructors
Stanford's deep generative models course has publicly posted slide decks and occasional lecture recordings covering score matching and SDEs. Quality varies by year; check cs236.stanford.edu or the course YouTube channel.

**Diffusion Models from Scratch** — Outlier / various YouTube educators
Multiple short-form series (search "diffusion models from scratch PyTorch") walk through DDPM math and implementation in 30-90 minutes. Useful for a quick visual overview before committing to longer material; no single canonical channel, so vet by view count and recency.

---

## Books / Surveys

**"Understanding Diffusion Models: A Unified Perspective"** — Calvin Luo (arXiv 2208.11970)
(Also listed under Blogs — it straddles both.) The closest thing to a textbook chapter on diffusion that is both rigorous and accessible. Covers ELBO derivation, DDPM, DDIM, classifier guidance, and score SDEs.

**Deep Learning** — Goodfellow, Bengio, Courville (deeplearningbook.org)
Does not cover diffusion directly, but provides the VAE / latent-variable / ELBO background assumed by every diffusion paper. Use as a reference when the probabilistic background feels shaky.

**"Score-Based Generative Modeling through Stochastic Differential Equations"** — Yang Song et al. (arXiv 2011.13456)
The landmark paper unifying NCSN and DDPM under an SDE framework. Dense but self-contained; read after Lilian Weng's blog and Calvin Luo's tutorial to see the full theoretical picture.
