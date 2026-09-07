---
name: verified-cascade-proposal
description: Proposed cascade v2 with grounding-check escalation trigger, plus simulated numbers from runs/ dumps
metadata:
  type: project
---

On 2026-09-03 I proposed "verified cascade" (cascade v2) for the RAG paper: keep
fused retrieval as stage 1, but widen the escalation trigger from "reader
abstained" to "abstained OR the generated answer does not appear verbatim in the
retrieved passages" (yes/no answers exempt, zero LLM cost). Simulated from the
per-question dumps in runs/ (cascade stages behave identically standalone, so
composition is exact): with grounding trigger, hybrid->hyde->closedbook scores
EM 0.3512 NQ / 0.6133 TriviaQA / 0.3498 HotpotQA / 0.2814 2Wiki / 0.0824 MuSiQue,
beating the abstention-only trigger on 4 of 5 sets; rerank->hyde->closedbook
recovers rerank's NQ collapse (0.2352 -> 0.3206). Simulation script:
scratchpad simulate_cascade.py (session scratchpad, may be gone). Caveat: unlike
abstention, the grounding trigger is NOT lossless (can discard a correct
parametric answer) - needs an ablation quantifying the discard rate. Note:
runs/cascade-* and runs/fused-* did not exist yet (cascade.go untracked, old
data removed as fake); README's cascade numbers had no backing runs.
