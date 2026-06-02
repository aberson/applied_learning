# Seed ideas — model distillation in the real workspace

Grounded applications of knowledge distillation across the `dev/` projects. Each idea names the real artifact it rides on. Terse reference tone.

## b2_project_goblin (strongest fit)

The costliest, riskiest part of goblin is `grade.py`: a 4-axis LLM-judge rubric run with **per-axis median-over-N** via concurrent `claude -p` subprocess fan-out (`cli.py` `run_suggest`, ThreadPoolExecutor). Every candidate suggestion eats N x 4 big-LLM calls. This is the #1 cost+risk.

- **Teacher -> student:** the existing `claude -p` judge -> a small local scorer (regression head per axis, or a single MLP over candidate embeddings). 
- **Flavor:** sequence-level / LLM-output distillation. The judge emits a structured per-axis verdict; the student learns to predict the median-over-N score directly.
- **Payoff:** collapse N x 4 LLM calls per candidate to one local forward pass. Removes the dominant latency and OAuth-billing cost; also de-risks the discrimination-guard flakiness (median-over-N exists precisely because single judge calls are noisy — a distilled student is deterministic).
- **First step:** the grade pipeline already persists suggestions under `brain/suggestions/`. Log every `(candidate text, per-axis median verdict, drop/keep)` tuple as training data — the **median-over-N is itself the denoised soft target**, which is exactly what you want as a distillation label. Once a few hundred graded candidates accumulate, fit a per-axis ridge/MLP over embeddings and compare its ranking against the LLM ranking from `rank.py` (mean of axis-medians). Keep the LLM judge as a periodic re-calibration teacher.

A second, weaker idea: distill `generate.py`'s candidate proposer too — but generation quality is the product, not a cost center, so leave it on the big model.

## toybox (strong fit)

Local-first, on a child **kiosk**, offline + privacy by design (`capability gate` already exists for offline degradation; Claude auth is OAuth-bearer + `urllib`, no SDK). Distillation is the canonical on-device play.

- **Intent classifier:** big Claude call (via the capability gate) -> tiny on-device intent model selecting among the 200 branching templates (`src/toybox/activities/templates/branching/`, ~50 per intent x 4 intents). 
- **Flavor:** logit / soft-target distillation. Teacher's class distribution over intents trains a small ONNX classifier (the stack already runs ONNX for audio).
- **Payoff:** intent routing works with zero network, sub-100ms, and never ships a child utterance off-device. Strengthens the capability gate's offline path from "degrade" to "fully local."
- **First step:** mine the existing activity-loop logs / `documentation/runs/` for (utterance -> chosen template/intent) pairs; if sparse, have the big model label a batch of held utterances to bootstrap soft targets, then train + export ONNX next to the faster-whisper / silero models in `data/models`.

Second idea: a small on-device **content/persona generator** distilled from the big model for step-card text, so approved activities render without a call. Honest caveat: content safety is non-negotiable, so keep the big model in the parent-approval loop and only distill the rendering of already-approved scripts.

## Alpha4Gate (strong fit)

The **Claude Advisor** is the top layer of the six-layer stack (Advisor -> Neural Engine -> Strategy -> ...). In-game advice from a big LLM is too slow for per-decision use.

- **Advisor -> policy/value head:** distill the Claude Advisor's guidance into a small fast head the Neural Engine can query in-loop. 
- **Flavor:** sequence-level / policy distillation — treat advisor outputs as expert action labels (or soft action distributions) over the same game-state features the PPO net already consumes.
- **Payoff:** in-game decisions stop needing a big-LLM round trip; the bot can use "advisor-flavored" guidance every tick instead of occasionally.
- **First step:** there is already a `decision_audit.json` / advisor-decision logging surface. Harvest (game state, advisor recommendation) pairs from logged games into a supervised set; train a small head, evaluate it through the existing evaluator / Elo ladder (`src/orchestrator/ladder.py`) against the advisor-on baseline.

- **Ensemble-to-single:** the project runs successive promoted versions (v0->v1->...->v4 via `bots/current`). Distill an **ensemble of promoted PPO nets** into one student net.
- **Flavor:** ensemble distillation (average the teacher policies' action distributions).
- **Payoff:** one net with ensemble-level robustness at single-net inference cost; cleaner to promote than juggling N checkpoints.
- **First step:** load the v2/v3/v4 checkpoints under `bots/v0/data/checkpoints/`, generate self-play states, train a student to match averaged logits, gate via the promotion pipeline.

## void_furnace (moderate fit)

Per-iteration LLM cost is dominated by the Planner -> Coder -> **Critic** loop (Claude Sonnet 4.6, OAuth). The Critic is the recurring judge cost.

- **Teacher -> student:** the production Critic -> a cheaper distilled judge for first-pass screening. 
- **Flavor:** sequence-level / LLM-output distillation on Critic verdicts (PASS/REJECT + rationale).
- **Payoff:** cut per-iteration cost; run the cheap judge on every coder diff, escalate only ambiguous cases to the full Critic.
- **First step:** Critic verdicts are already persisted (`runs/<ulid>/`, retrospectives). Mine (coder-diff, critic verdict) pairs as labels. **Hard constraint:** the **holdout principle** (critic never sees coder artifacts; enforced by signature + unit test) must survive distillation — the student must be trained and served on the same redacted view, or it silently breaks the invariant. Treat that test as the gate.

## sandtable (weak fit — skip)

Small voice toy (songs/jokes/stories/facts), no trained model and no per-call cost center visible in CLAUDE.md; the ≥99%-accuracy / 100%-age-appropriate invariants argue for keeping the big model, not distilling. Revisit only if it grows an on-device latency requirement.

## Cross-cutting

- **Any `claude -p`-heavy judge/critic -> cheap local model.** goblin's grader and void_furnace's Critic are the same pattern: an LLM scoring artifacts N times. The reusable recipe — log `(input, denoised verdict)`, distill into a local scorer, keep the LLM as periodic re-calibration teacher — generalizes to any `_shared/score_skill_composite.py`-style grader in the skill pipeline (e.g. `/skill-iterate`, `/skill-evolve` scoring loops). The median-over-N pattern is the gift: it hands you denoised soft targets for free.
- **Ensemble-to-single for any promoted-model line.** Alpha4Gate's v0->v4 promotion chain is the prime case, but any project that keeps multiple trained checkpoints can distill them into one robust student at single-net cost.
- **Cascade, don't replace.** For risk-bearing judges (content safety in toybox, holdout in void_furnace, ranking fidelity in goblin), run the distilled student as a cheap first pass and escalate uncertain cases to the teacher — captures most of the cost win without surrendering the invariant the big model guarantees.
