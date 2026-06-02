# Diffusion models — seed ideas (mentoring)

Grounded ideas for applying diffusion models in real workspace projects. Each tied to an
artifact that already exists in the project (read the project's `CLAUDE.md` + runbooks before
starting). Terse reference tone; pick the strong fits first.

## toybox

toybox already ships a production diffusion pipeline (SD 1.5 + LCM-LoRA @ 512², 4-step,
[`documentation/operator/image-gen-runtime.md`](../../../../toybox/documentation/operator/image-gen-runtime.md))
and Phase P added IP-Adapter Plus for subject identity. So these are extensions of a live
system, not greenfield.

- **On-brand toy-action sprites with stronger identity.** The pipeline already conditions on a
  toy's reference photo (`data/images/toys/<id>.jpg`) via IP-Adapter Plus to render the 10
  `ACTION_SLOTS` sprites. (a) Improve identity preservation / consistency across slots.
  (b) IP-Adapter Plus tuning — image-encoder choice, scale weight, prompt blend; the workspace
  IPA gotchas (explicit `image_encoder`, no attention_slicing, `vae.enable_slicing()`) already
  apply. (c) First step: A/B current IPA scale vs. a sweep on the 3-toy smoke probe
  (`python -m toybox.image_gen --probe <toy_id> --slot idle`), score by recognizability.
- **ControlNet pose for the per-element Professor Iridia sprites.** `scripts/generate_element_sprites.py`
  renders the persona holding an element orb, 118 elements, with a deterministic per-element
  seed. (a) Lock pose/composition across all 118 so the panel-overlay always lands cleanly.
  (b) ControlNet (OpenPose or scribble) on one canonical Iridia pose. (c) First step: extract a
  pose map from the best existing committed anchor sprite, condition one re-roll, compare framing.
- **Inpainting for sprite cleanup instead of Tier-C fallback.** When a toy renders an unusable
  slot, the current fallback deletes the dir and drops to the composite template. (a) Mask + regen
  just the broken region (extra limb, garbled face). (b) SD 1.5 inpainting checkpoint reusing the
  same LCM scheduler. (c) First step: hand-mask one known-bad probe output, inpaint, confirm it
  beats the Tier-C composite for that toy.

## sandtable

sandtable converts a child's speech into personalized songs/jokes/stories/facts (+1 category),
currently a Python text/audio pipeline with no image surface (`docs/project_overview.md`). The
≥99% accuracy / 100% age-appropriate invariants are non-negotiable.

- **Story / song illustration.** Generate one illustration per personalized story or song.
  (a) Text-to-image from the already-generated story text (auto-derive a prompt from the
  story's subject). (b) SD 1.5 + a cartoon checkpoint — reuse toybox's exact validated config
  (4-step LCM @ 512²) rather than re-deriving. (c) First step: a standalone script that takes one
  canned story string and emits a PNG; no integration yet.
- **Recurring on-brand character via IP-Adapter.** If sandtable grows a mascot, keep it visually
  consistent across every generated illustration. (a) Condition each render on one reference
  mascot image. (b) IP-Adapter Plus (the toybox identity mechanism, lifted whole). (c) First step:
  pick/draw one reference, render the same mascot in 3 different story scenes, eyeball consistency.
- **Safety note (load-bearing):** sandtable's 100%-age-appropriate invariant must extend to
  generated imagery — gate prompts through the same content filter as text, and human-review the
  illustration model's output before any kiosk surface. Do not ship image-gen until that gate
  exists (pair the unsafe capability with a startup/runtime safety check, per workspace security
  rules).

## Alpha4Gate

Honest framing: **text-to-image does not fit** an SC2 bot. Where diffusion genuinely applies is
generative modeling for control and data, not pixels.

- **Diffusion policy for action generation.** The bot has a PPO neural policy
  (`bots/v0/learning/`) over its command system. (a) Replace/augment the policy head with a
  conditional diffusion model that generates an action (or short action sequence) conditioned on
  game state — diffusion policies handle multimodal action distributions PPO collapses.
  (b) Diffusion-policy (DDPM/DDIM over the action vector, state as conditioning). (c) First step:
  offline — train a diffusion policy on logged `transitions` (the SQLite `training.db` +
  `decision_audit.json`) via behavior cloning, evaluate action-distribution fidelity before any
  live wiring. Honest caveat: this is a research bet, not a quick win; gate it on the same fitness
  noise floor the evolve loop uses.
- **Generative data augmentation for rare states.** The bot wins at difficulty 1-3, struggles at
  4-5 — partly a rare-state coverage problem (the `winprob_heuristic` / give-up trigger fires on
  states that are under-sampled). (a) Train a diffusion model over the state-feature vectors to
  synthesize plausible rare/late-game states for offline policy training. (b) Tabular/vector
  diffusion (not image). (c) First step: fit on the `transitions` feature columns, validate that
  sampled states are in-distribution (no impossible resource/supply combos) before feeding any
  trainer. Caveat per the workspace small-sample-validation rule: instrument and verify the
  augmentation actually shifts difficulty-4-5 win rate, don't assume.

## void_furnace

**Poor fit.** void_furnace is an autonomous code-gen factory (Planner→Coder→Critic over text);
no image, audio, or continuous-control surface. Diffusion adds nothing here — skip it.

## b2_project_goblin

**Poor fit.** goblin is a markdown/LLM-judge project-improvement engine over text atoms; no
generative-media or control surface. Skip diffusion.

## Cross-cutting

- **Shared asset-generation pipeline.** toybox and sandtable both want cartoon kid-art from the
  same SD 1.5 + LCM + IP-Adapter stack. Factor the validated config (the toybox canonical block)
  into one small reusable module so sandtable doesn't re-derive the diffusers gotchas (offload
  crash, attention-slicing, VAE slicing, IPA image-encoder). One source of truth for the pipeline
  config, per the workspace duplicate-shape-constants rule.
- **Synthetic data generation for ML projects.** Vector/tabular diffusion as a general tool:
  Alpha4Gate rare-state synthesis above is the concrete instance, but the same technique seeds
  any future model needing balanced training data from a skewed real log.
- **Capability-gated, offline-first by default.** Every diffusion surface above must degrade
  gracefully without a GPU — toybox already models this (Tier B diffusion → Tier C composite →
  persona-only). Any new project adopting diffusion should copy that three-tier capability gate
  rather than hard-requiring CUDA, keeping the local-first/family-private posture intact.
