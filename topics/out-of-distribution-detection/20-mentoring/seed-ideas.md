# Seed ideas — OOD detection in the real workspace

Concrete places to wire an OOD signal into projects that already exist, mapped to the
technique that produces it. Notebook order: MSP → [02](../70-handson/notebooks/02-msp-baseline-mnist.ipynb),
ODIN/energy → [03](../70-handson/notebooks/03-odin-and-energy.ipynb), Mahalanobis/KNN →
[04](../70-handson/notebooks/04-mahalanobis-and-knn.ipynb), ensembles/entropy →
[05](../70-handson/notebooks/05-ensembles-uncertainty.ipynb).

## Start here

**switchboard's energy-defer.** `local_judge` already defers to Claude on parse failure or
outage — that's a fail-safe skeleton with no confidence signal in it. Add an energy or
max-softmax score ([02](../70-handson/notebooks/02-msp-baseline-mnist.ipynb) /
[03](../70-handson/notebooks/03-odin-and-energy.ipynb)) on general-35b's output logits and defer
*proactively* when the prompt looks OOD for the local model, not just when it crashes. Smallest
diff, existing fallback path, no new infra — the single most tractable prototype in this list.

## void_furnace

Planner→Coder→Critic factory currently gates the open-model tier (llama.cpp coder-30b) on a
binary harness flip.

- **Per-task safety gate**: energy/max-softmax on the coder's logits per build to decide cheap-tier
  vs. OAuth-Sonnet fallback — replaces the all-or-nothing "flip HELD until clean soak" with a
  quantitative per-task gate. Notebooks [02](../70-handson/notebooks/02-msp-baseline-mnist.ipynb)/[03](../70-handson/notebooks/03-odin-and-energy.ipynb).
- **5th verdict bucket**: Mahalanobis/KNN distance ([04](../70-handson/notebooks/04-mahalanobis-and-knn.ipynb))
  of an incoming issue embedding against `scripts/readiness_bench/`'s labeled set — issues far from
  any past success get an "OOD → escalate" verdict alongside INTRACTABLE/BLOCKED/ENRICHABLE in
  `readiness/judge.py`.
- **Judge disagreement**: predictive entropy/disagreement across the existing k=3 median LLM-judge
  scoring — high disagreement on governance_compliance is itself an OOD signal, route to human
  ([05](../70-handson/notebooks/05-ensembles-uncertainty.ipynb)).

## switchboard

- Energy/max-softmax proactive defer (see Start here).
- Predictive entropy on the local verdict as an accept/relay threshold before trusting a local
  judge result over relaying to Claude ([05](../70-handson/notebooks/05-ensembles-uncertainty.ipynb)).

## Alpha4Gate

Rule-based strategy + PPO policy + Claude advisor.

- Mahalanobis/KNN on PPO policy features vs. training states ([04](../70-handson/notebooks/04-mahalanobis-and-knn.ipynb))
  to detect a novel game state (unseen opponent build) and drop from AI-Assisted to Human/Hybrid
  command mode, or invoke the Claude Advisor instead of the Neural Engine.
- Pair an OOD score with `winprob_heuristic.py`/`give_up.py`: resign/abstain when the state is far
  from training distribution, not merely low win_prob.
- Deep-ensemble disagreement across ladder-promoted versions (v0→v4) as an OOD signal on the
  current state ([05](../70-handson/notebooks/05-ensembles-uncertainty.ipynb)).

## toybox

Passive listening → activity suggestion → child kiosk.

- Max-softmax/energy on the intent classifier (request_play/request_story/request_activity/
  boredom) to abstain and ask the parent when an utterance is far from the ~1360 branching
  templates, instead of forcing a wrong intent through the capability gate.
- Mahalanobis/KNN distance in the Phase-X local-CLIP room-import match to reject "this photo isn't
  a recognizable room."

## b2_project_goblin

`grade.py`'s per-axis median-over-N + discrimination guard is already ensemble-flavored. Add
predictive-entropy/judge-disagreement ([05](../70-handson/notebooks/05-ensembles-uncertainty.ipynb))
as an explicit OOD drop for suggestions where judges wildly disagree — the input is outside the
rubric's reliable range.

## dev-observatory

No inference of its own, so no honest OOD hook — `switchboard_panel.py` could only *surface* a
per-endpoint OOD/confidence score computed by switchboard; the OOD logic itself belongs there.
