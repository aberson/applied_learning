# Model Distillation — Frontier / Applied / Surveys

LLM distillation, transformers, improved objectives, and reviews.

---

## Surveys

**[Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)**
Gou, Yu, Maybank, Tao — 2021
Comprehensive survey covering response-based, feature-based, and relation-based distillation methods across vision and NLP. Essential orientation before engaging with any sub-area of the literature.

---

## LLM / sequence-model distillation

**[Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)**
Hsieh et al. — 2023
Uses chain-of-thought rationales from a large teacher as an additional supervision signal, enabling smaller students to outperform the teacher on several benchmarks with far fewer labeled examples. Shows that rationale-augmented distillation beats both standard fine-tuning and vanilla KD under data constraints.

**[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)**
Gu, Dong, Wei, Huang — 2023
Replaces the standard forward KL objective with reverse KL to avoid mode-averaging artifacts in auto-regressive generation, training students via policy-gradient optimization. Demonstrates consistent generation quality improvements over forward-KL baselines across multiple LLM families.

**[GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models](https://arxiv.org/abs/2306.13649)**
Agarwal et al. — 2023
Frames sequence-model distillation as a unified on-policy/off-policy learning problem, showing that student-generated sequences (not just teacher sequences) must appear in training to avoid compounding errors. Provides a principled interpolation between on-policy and off-policy regimes applicable across diverse divergence measures.

**[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)**
Jiao et al. — 2019
Introduces two-stage distillation (general pre-training distillation + task-specific distillation) with layer-to-layer attention and hidden-state alignment. Achieves ~96% of BERT-base performance at 28% of its parameters, establishing a strong transformer-distillation baseline.

**[Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)**
Tang et al. — 2019
Transfers BERT's soft-label supervision into shallow BiLSTM and single-layer networks, recovering most of BERT's NLP task performance in architectures orders of magnitude smaller. Highlights that soft targets alone carry enough signal to close a large architecture gap.

---

## Transformers & improved objectives

**[Training data-efficient image transformers & distillation through attention (DeiT)](https://arxiv.org/abs/2012.12877)**
Touvron et al. — 2020
Introduces a distillation token alongside the class token in ViT, letting the student learn from a convolutional teacher's hard decisions via attention. Enables training competitive vision transformers from scratch on ImageNet without extra data, removing the previous data-scale requirement.

**[Decoupled Knowledge Distillation (DKD)](https://arxiv.org/abs/2203.08679)**
Zhao, Cui, Song, Qiu, Liang — 2022
Decomposes the classical KD loss into target-class KD (TCKD) and non-target-class KD (NCKD), showing the two terms have opposing and previously conflated effects. Decoupling them with independent weights yields consistent gains over KD and feature-based methods across standard benchmarks.

**[Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393)**
Mirzadeh et al. — 2019
Identifies a capacity gap regime where large teacher-student size differences degrade transfer, and introduces intermediate teacher-assistant models to bridge the gap. Provides both empirical evidence and analytical justification for multi-step distillation chains.

**[Model compression via distillation and quantization](https://arxiv.org/abs/1802.05668)**
Polino, Pascanu, Alistarh — 2018
Combines quantization-aware training with knowledge distillation in a joint objective, using the full-precision teacher to guide the quantized student. Shows that distillation recovers accuracy lost to aggressive quantization, making the two compression axes complementary rather than independent.
