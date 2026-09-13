# Rag
* Setup
* Health check the embedder
* Ingest data
* Measure corpus coverage
* Run benchmark
* The proposed architecture
* Diagnose retrieval failures
* Compare runs
* Check ingested data
* Clean collection
* Dataset
* Groups
* RAG Measurements 
* Generation Measurements
* Bibliography

> The `-dump` schema changed: it now records what was retrieved, what the
> architecture decided and what it cost. Runs made before that are not readable
> by `cmd/compare` and have to be regenerated.

## Setup

```sh
docker compose up -d qdrant vllm-embed vllm-llm vllm-rerank
```

## Health check the embedder

```sh
curl -s http://localhost:8001/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"bge-base-en-v1.5","input":["hello world"],"truncate_prompt_tokens":512}' | jq '{dim:(.data[0].embedding|length), tokens:.usage.prompt_tokens}'
```

## Ingest data

Two indexes over the same corpus and the same passage ids. They are separate
collections so that adding lexical search never means re-embedding the 21M
passages the dense one already holds.

```sh
make ingest          # ~3h, GPU: embeds the corpus into ragbench-wiki18
make ingest-sparse   # ~25min, no GPU: BM25 index into ragbench-wiki18-bm25
```

Both are restartable. An interrupted run prints the corpus line to resume from;
re-run the command by hand with `-skip N`. Upserts are keyed by corpus id, so
re-running an overlapping range is safe.

## Measure corpus coverage

A Recall figure is unreadable without the ceiling the corpus imposes. Only 72.6%
of the gold articles 2WikiMultihopQA annotates exist in wiki18\_100w at all, so a
perfect retriever would score 0.726 there and NaiveRAG's measured 0.354 is 49%
of what is achievable, not 35% of it.

```sh
make coverage        # ~4min, no GPU, can run while the card is busy
```

NaturalQuestions and TriviaQA annotate no gold documents, so there is no
coverage to measure for them.

## Run benchmark

Every run is a target, so there is one definition of each experiment and the
paper's numbers cannot drift from the commands that produced them. Run one, or
run the lot:

```sh
make bench           # every architecture below, cheapest first
make naive-rag       # just this one, on all five question sets
```

| target | architecture | question sets | cost per question |
|---|---|---|---|
| `closedbook` | no retrieval at all - the floor every other row is read against | all 5 | 1 LLM call |
| `naive-rag` | one query, top 5 passages | all 5 | 1 |
| `naive-rag-ten` | top 10 - matched-budget control for IRCoT and CRAG | multi-hop 3 | 1 |
| `naive-rag-twelve` | top 12 - matched-budget control for neighbour expansion | multi-hop 3 | 1 |
| `bm25` | lexical retrieval only, no embedding model in the loop | all 5 | 1 |
| `hybrid` | dense + BM25 fused by Reciprocal Rank Fusion | all 5 | 1 |
| `hyde` | search with a drafted passage instead of the question | all 5 | 2 |
| `rerank` | 100 candidates reordered by a cross-encoder | all 5 | 1 + 1 cross-encoder pass |
| `neighbour` | each hit joined by its neighbours in the same article | multi-hop 3 | 1 |
| `crag` | grade the retrieval, rewrite and re-search when it is poor | all 5 | 3 |
| `crag-ten` | the same at a 10-passage budget | multi-hop 3 | 3 |
| `ircot` | interleave one sentence of reasoning with one retrieval | all 5 | 2-6 |
| `ircot-oneshot` | the same with one worked example in the prompt, as FlashRAG runs it - outside `make bench`, see below | all 5 | 2-6 |
| `adaptive` | one call routes to closedbook, naive or ircot | all 5 | 2-7 |
| `cascade` | **the proposed architecture** - see below | all 5 | ~1.3 |

Every recipe is a plain `go run` line, so a single question set can be run by
copying the one line out of the Makefile and editing it - use a scratch `-dump`
path if you add `-limit` for a smoke test, or the real files get overwritten
with a sample.

## The proposed architecture

`cascade` is what the measurements above argue for, in two parts.

**Fused retrieval.** Cross-encoder reranking is the strongest single retrieval
improvement measured here (+0.0510 Exact Match on HotpotQA, +0.0388 on
2WikiMultihopQA) and it also fails catastrophically on NaturalQuestions
(-0.1001, answer-in-context 0.7267 -> 0.5097). The cause is that a
cross-encoder matches what a question is *asking for* while a bi-encoder
matches what it is *asking about*: for "where is the tv show the curse of oak
island filmed" it promotes the passages that state a filming location - for
Stake Land, The Island and Come Outside - and pushes out the one naming the
right show. Fusing the dense, lexical and cross-encoder rankings with RRF keeps
both signals, at the cost of one extra lexical search on the CPU. Measured, the
fusion is a compromise rather than a free lunch. Stage 1 on its own (recovered
from `runs/cascade-*.jsonl`, see Ablations) scores 0.3253 Exact Match on
NaturalQuestions against rerank's 0.2352 and naive's 0.3353, and 0.3514 on
HotpotQA against rerank's 0.3757 and naive's 0.3249: it avoids the collapse but
gives back part of the gain.

**Abstention-triggered escalation.** When the reader declines to answer, it has
already seen the retrieved passages and reported that they do not support an
answer - a *posterior* signal, unlike `adaptive`, which asks the generator to
predict difficulty from the question alone and gains +0.0004 Exact Match for it
on 2WikiMultihopQA. The refusal is also lossless: across all 887,917 (question,
architecture) pairs measured here there is no case where an abstention scores
Exact Match 1, because
an abstention is a sentence and the gold answers are short spans. Escalating a
declined question therefore cannot discard a correct answer. Only the declined
questions pay for the next stage, which is why the whole thing runs at about
1.3 generation calls per question against 3.0 for CRAG and 2.3 for IRCoT.

### Generating and comparing its runs

`cascade` is part of `make bench`, so `make bench` (or `make cascade` alone)
produces `runs/cascade-<set>.jsonl` and `paper/generated/cascade-<set>.gen.tex`. It needs
both collections - the dense one and the BM25 one - and the reranker
container, because stage 1 fuses all three. The fragment carries, on top of the
usual metrics, how many questions each stage answered
(`\Cascade<Set>StageFused`, `StageHyde`, `StageClosedbook`) and the mean
number of stages run. `make compare` then writes the two paired tests the
paper reads it from, `paper/generated/cmp-rerank-cascade-<set>.gen.tex` and
`paper/generated/cmp-naive-cascade-<set>.gen.tex`. Every cascade row in the per-question
dump names the stage that answered, so per-stage Exact Match is a one-line
aggregation over `stage`.

### The confidence trigger

The abstention trigger is lossless but blind to a wrong answer given without
hesitation, and that is where the remaining headroom is: an oracle choosing
per question between naive, rerank and closed-book scores 0.403 on
2WikiMultihopQA against the cascade's 0.278, of which the abstention collects
1.6 points. The second signal is the generator's own certainty - the mean
log-probability of the answer's tokens, which vLLM reports for free and every
run now records as `mean_logprob` in the dump. With `-cascade-min-logprob`
the cascade also escalates an answer below that threshold.

It is not lossless: a correct answer given with low confidence is discarded
and only sometimes recovered. So the threshold is a hyperparameter, and it is
chosen on held-out questions, never on the test sets:

```sh
make cascade-tune                 # ~1h: stage runs on 1000 train questions per set, then the offline sweep
make cascade-logprob TAU=-0.5     # the test-set run at the chosen threshold, its comparisons, its result table
```

`cascade-tune` runs `fused`, `hyde` and `closedbook` on the same 1000 sampled
questions of each train split and then `cmd/sweep` composes them question by
question at every threshold, printing Exact Match, calls per question, how
many stage-1 answers were escalated, how many of those were correct
(discarded) and how many a later stage did not get back (missed). One value
is chosen for all sets; put it in `TAU`. The test run writes
`runs/cascade-lp-<set>.jsonl` and is compared with the abstention-only
cascade, naive and rerank, and rendered as its own summary table
(`paper/generated/result-lp.gen.tex`).

Measured, it does not pay. The sweep over 4,988 train questions put the best
threshold at -0.15 for +0.0018 Exact Match (McNemar p = 0.42) at +0.25 calls.
On the test sets at that threshold the trigger fires on 9-15% of questions and
the paired difference against the abstention-only cascade is -0.0008 on
NaturalQuestions, +0.0021 on TriviaQA, -0.0020 on HotpotQA, -0.0037 on
2WikiMultihopQA (the one significant difference, a loss) and 0.0000 on
MuSiQue, at 0.2-0.36 more generation calls per question. The greedy reader is
nearly as confident when wrong as when right, so its log-probability carries
almost no information the abstention did not already carry. `cascade` stays
the proposed row; `cascade-lp` is the ablation that says so.

### The summary table

`make result` renders `paper/generated/result.gen.tex`: the proposed architecture against
every baseline on every set, one cell per pair, each cell the paired Exact
Match difference marked as a win (bold), a loss (underlined) or a tie (plain).
The test is McNemar's, as in `cmd/compare`, but Holm-adjusted over every cell
of the table at once, because the sentence the table supports - "wins W,
loses L, ties T of P pairs" - is one family of tests. That is stricter than
the per-pair figures in `paper/generated/cmp-*.gen.tex`, and a few narrow wins there
are ties here.

### The cost table

`make cost` renders `paper/generated/cost.gen.tex`: generation calls, prompt tokens,
completion tokens and passages per question for every architecture, averaged
over the sets it ran on, plus the prompt-token ratio to NaiveRAG. It reads the
dumps, so it is tokens and calls only - the comparable cost. The throughput in
the per-run fragments is not in it on purpose: at `-concurrency 48` it
measures the queue, and runs made at different times against a shared card
differ by a factor of two at identical token counts. What the table also does
not count is the work outside the generator - the cross-encoder pass over 100
candidates (rerank, fused, cascade) and the BM25 search over 21M passages on
the CPU (bm25, hybrid, fused, cascade) - which is why hybrid runs a third
slower than naive at the same token cost.

### Ablations

The architecture is one target, `cascade`, and one row in the main table. The
ablations are one more target, outside `make bench` because none of them is a
competing method. Every line in it is the same loop (`pkg/rag/cascade.go`)
with one element switched through `-cascade-stages`; the fused retriever
itself lives in `pkg/rag/fused.go`.

```sh
make cascade-ablations   # after make cascade; ~4 benchmark runs per set plus comparisons
```

| run | stages | answers |
|---|---|---|
| `fused` | fused only, no escalation | what the escalation adds |
| `cascade-rr` | rerank -> hyde -> closedbook | what the fusion adds |
| `cascade-naive` | naive -> hyde -> closedbook | the cheapest possible cascade |
| `cascade-hybrid` | hybrid -> hyde -> closedbook | fusion without the cross-encoder |

The target ends with the paired comparisons of each variant against `cascade`
(`paper/generated/cmp-<variant>-cascade-<set>.gen.tex`) and of `fused` against `rerank`.

The `fused` run is optional for Exact Match. Stage 1 answers every question in
the cascade, and every escalated question is one where stage 1 abstained -
which scores Exact Match 0 by construction - so stage-1 Exact Match is exactly
the sum over the rows of `runs/cascade-*.jsonl` whose `stage` is `fused`. The
run exists for the metrics that are not recoverable that way: stage 1's own
Recall, MRR and answer-in-context.

What the dumps already say about these variants, before the runs exist:
composing standalone runs question by question reproduces the cascade's later
stages on 95-100% of questions, and that composition puts rerank-first ahead
on HotpotQA (+0.021), level on TriviaQA, 2WikiMultihopQA and MuSiQue, and
-0.052 behind on NaturalQuestions, where reranking answers wrongly rather than
declining. No first stage wins everywhere; fused has the best mean and the
smallest worst-case loss, which is its case.

### Reading its row

Three things to keep in mind:

- **Compare it on Exact Match, not on Recall or MRR.** Questions that reach the
  closed-book stage end with no passages in context at all, so its retrieval
  metrics are averages over a mixture and are not comparable with a pure
  retrieval system's.
- The losslessness of the trigger is exact for Exact Match and only approximate
  for token-level F1, where an abstention averages 0.0495 and exceeds 0.5 for
  0.09% of questions.
- **Its abstention rate is the last stage's.** Against a single-stage system
  the "EM (both answered)" row of `cmd/compare` is therefore computed over a
  different population, and on four of the five sets it is at or below zero
  against rerank. The cascade's gain is in the questions the baseline declined,
  not in the questions both answered - which is what the design predicts, and
  what the paper has to say.

### Where it stands

Paired Exact Match differences from `paper/generated/cmp-*-cascade-*.gen.tex` (Holm-adjusted
p on the primary endpoint), generation calls per question in brackets:

| set | vs NaiveRAG [1.0] | vs Rerank [1.0] | cascade calls |
|---|---|---|---|
| NaturalQuestions | +0.0064 (n.s.) | +0.1066 | 1.44 |
| TriviaQA | +0.0333 | +0.0071 (p=0.09) | 1.16 |
| HotpotQA | +0.0334 | **-0.0174** | 1.22 |
| 2WikiMultihopQA | +0.0497 | +0.0108 | 1.32 |
| MuSiQue | +0.0244 | +0.0029 (n.s.) | 1.58 |

It is not the best system on every set: HyDE beats it on NaturalQuestions
(0.3476 against 0.3417 at 2.0 calls), rerank on HotpotQA, and CRAG at a
10-passage budget on MuSiQue (0.0877 at 3.0 calls). What it is is the only
architecture measured here that is never worse than NaiveRAG and never
catastrophic, at 1.2-1.6 calls.

## Is the IRCoT baseline under-prompted

FlashRAG runs IRCoT with one worked example in the prompt and two iterations,
and reports it gaining +6.2 F1 over standard RAG on HotpotQA and +11.4 on
2WikiMultihopQA. The zero-shot loop here gains +1.8 and +1.9. A reviewer will
ask whether the baseline the cascade is read against was handicapped, so the
one-shot variant is its own row:

```sh
make ircot-oneshot   # runs/ircot1-<set>.jsonl, compared with ircot, naive10 and cascade
```

The example is one hand-written two-hop question (`rag.IRCoTDemonstration`),
identical for every set. The dumps already say where the small gain comes
from: the `reasoning_steps` field shows the zero-shot loop declaring it can
answer after the first round - five passages, the same context as NaiveRAG -
on 80.4% of HotpotQA questions, 70.3% of 2WikiMultihopQA and 67.1% of MuSiQue.
Whatever the example does to the score, it has to do it by changing that
fraction. If the row beats the zero-shot one it becomes the IRCoT baseline and
`adaptive`, which uses IRCoT as its multi-hop branch, has to be re-run.

## A second reader

Everything the cascade claims rests on how the reader behaves when the
passages are irrelevant - it declines, and the decline is what triggers the
escalation. One model is not evidence that readers in general do that, so the
rows the claims stand on are repeated with Llama-3.1-8B-Instruct, which is
also the generator FlashRAG's published numbers use. The index and the
questions stay the same; only the generator changes.

```sh
docker compose stop vllm-llm && docker compose up -d vllm-llm-llama   # both do not fit on the card at once
make reader                                                           # ~20h: closedbook, naive, rerank, fused, cascade on all 5 sets
docker compose stop vllm-llm-llama && docker compose up -d vllm-llm   # back to the main reader
```

The service pulls `NousResearch/Meta-Llama-3.1-8B-Instruct`, a byte-identical
mirror of Meta's gated repository, so no Hugging Face token is needed; cite
the Meta model in the paper. The runs are
written as `runs/<architecture>-llama-<set>.jsonl`, their fragments as
`paper/generated/<architecture>-llama-<set>.gen.tex` with `Llama` in the command names,
the six comparisons that matter as `paper/generated/cmp-*-llama-*.gen.tex`, and the
summary table as `paper/generated/result-llama.gen.tex`. Another model is
`make reader READER_MODEL=... READER_URL=... READER_TAG=... READER_NAME=...`.

## Diagnose retrieval failures

Splits the naive baseline's retrieval failures into causes that call for
different fixes - wrong slice of the right article, right article ranked too
low, article unreachable by this query, no gold article stating the answer,
article absent from the corpus - and reports what choosing per question between
retrieval and closed-book would be worth.

```sh
make diagnose
```

It needs `runs/naive-<slug>.jsonl` and `runs/closedbook-<slug>.jsonl`, so run
`make closedbook naive-rag` first.

## Compare runs

Paired significance tests over the per-question dumps: McNemar on the binary
metrics, Wilcoxon on the continuous ones, a paired bootstrap interval for the
size of each difference, and a Holm adjustment across the family because a
dozen tests on one pair of runs will produce a small p by chance alone.

```sh
make compare
```

| pair | why this control |
|---|---|
| closedbook vs naive | is retrieval worth anything at all on this dataset |
| naive vs bm25, hybrid, hyde, rerank | all keep the 5-passage budget, so the difference is which passages were chosen |
| naive10 vs ircot and crag10 | these put more passages in context, so the baseline is given the same number |
| naive12 vs neighbour | neighbour puts 10.7-12.3 passages in context |
| naive vs adaptive, ircot vs adaptive | a router is only interesting against the branches it picks between |
| ircot vs ircot-oneshot, naive10 vs ircot-oneshot, ircot-oneshot vs cascade | is the zero-shot IRCoT under-prompted; run by `make ircot-oneshot` |
| rerank vs cascade | the proposed architecture against the strongest single baseline |
| naive vs cascade | the proposed architecture against the standard baseline |
| each ablation vs cascade, rerank vs fused | run by `make cascade-ablations`, not by `make compare` |

`make all` runs the benchmarks, the diagnosis, the comparisons, the coverage,
the summary table and the cost table in order.

## Check ingested data

```sh
curl -s http://localhost:6333/collections/ragbench-test | jq '.result.points_count'
```

## Clean collection

```sh
curl -s -X DELETE http://localhost:6333/collections/ragbench-test
```

## Foundation
* Get the nearest pieces the same article (28–42% improvement). If the search result is an article, also include the piece before and after.
* Add keyword search (12–25% improvement). Cases where search is found by exact keyword match catch other results.
* Divide the question into steps (improves 11–26%). The impossible cases like "mother of the dicector of movie X". This is what IRCoT does so it wins

## Dataset
[RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus)
[Natural Questions (nq)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/nq)
[TriviaQA](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/triviaqa)
[HotpotQA](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/hotpotqa)
[2WikiMultihopQA](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/2wikimultihopqa)
[MuSiQue](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/musique)

## Groups
* Self-assesment
  * Self-RAG
  * CRAG (Corrective RAG) 
  * Adaptive-RAG
* Agentic
  * IRCoT
  * ReAct
  * ReWOO / LLMCompiler
  * Reflexion
* Structured
  * GraphRAG
  * HippoRAG 2
  * HoppRAG
  * RAPTOR
* Routing
  * RouteRAG
  * AT-RAG
  * CQC-RAG

## RAG Measurements
  * Recall in context: the fraction of a question's gold articles that appear among the passages actually placed in the generator's context. Not "Recall@5": IRCoT accumulates passages over several rounds and puts more than `-top-k` in context, so the mean context size is reported beside it and the matched-budget runs (`-top-k 10`, `-top-k 12`) are the controls that make the comparison fair.
  * Recall ceiling: the fraction of gold articles that exist in the corpus at all, measured by `cmd/coverage`. It is the maximum Recall in context can take, and it is well below 1 (0.726 on 2WikiMultihopQA, 0.899 on HotpotQA, 0.887 on MuSiQue).
  * MRR (Mean Reciprocal Rank): the average of the reciprocal ranks of the first relevant document for each query.
  * Answer in context: whether a gold answer appears as a run of whole tokens in a retrieved passage. Skipped for yes/no questions, which have no answer span in the supporting text.
  * Article titles are compared after Unicode normalisation; the corpus and the question sets disagree on whether an accented letter is stored precomposed, and a byte comparison loses 4.5% of 2WikiMultihopQA's gold articles to that alone.

## Generation Measurements
  * Exact Match (EM): does the response match the answer exactly.
  * F1 (token-level): the F1 score at the token level.
  * Abstention rate: how often the system declined instead of answering. Exact Match cannot tell a refusal from a wrong guess, and a reader holding irrelevant documents refuses while a closed-book reader guesses - which is what makes ClosedBook appear to beat NaiveRAG on 2WikiMultihopQA. EM is therefore also reported over the questions the system did not decline.
  * LLM calls and tokens per question: the cost measure that is comparable across hardware, unlike latency.
  * Latency: only meaningful at `-concurrency 1`; above it the per-question timing includes queueing, and the generated fragment omits the command entirely so a table cannot quote it by accident.
  * Faithfulness/groundedness and Answer Relevance: not implemented.

## A note on naming

The `crag` architecture is not the full method and is labelled "CRAG (offline,
corpus-only)" in the paper: there is no web search (a query rewrite over the
same corpus replaces it, which bounds it by what the corpus contains) and no
knowledge refinement step. The `adaptive` router likewise classifies with the
generator instead of the paper's trained classifier. Both deviations keep the
corpus, retriever and generator identical across architectures, which is what
makes the comparison controlled - but they have to be named.

## Bibliography
- Agentic Retrieval-Augmented Generation: A Survey — https://arxiv.org/abs/2501.09136
- AgenticRAG-Survey (companion repo) — https://github.com/asinghcsu/AgenticRAG-Survey
- FlashRAG (paper) — https://arxiv.org/abs/2405.13576
- FlashRAG (repo) — https://github.com/RUC-NLPIR/FlashRAG
- FlashRAG datasets (Hugging Face) — https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
- Self-RAG — https://arxiv.org/abs/2310.11511
- CRAG (Corrective RAG) — https://arxiv.org/abs/2401.15884
- Adaptive-RAG — https://arxiv.org/abs/2403.14403
- HippoRAG 2 — https://arxiv.org/abs/2502.14802
- GraphRAG — https://arxiv.org/abs/2404.16130
- RAPTOR — https://arxiv.org/abs/2401.18059
- IRCoT — https://arxiv.org/abs/2212.10509
- ReAct — https://arxiv.org/abs/2210.03629
- ReWOO — https://arxiv.org/abs/2305.18323
- LLMCompiler — https://arxiv.org/abs/2312.04511
- Reflexion — https://arxiv.org/abs/2303.11366
- MultiHop-RAG — https://arxiv.org/abs/2401.15391
- RAG vs. GraphRAG: A Systematic Evaluation — https://arxiv.org/abs/2502.11371
- RAGRouter-Bench — https://arxiv.org/abs/2602.00296
- MobilityBench — https://arxiv.org/abs/2602.22638
- From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review — https://arxiv.org/abs/2504.19678
- SEAL-RAG (Mitigating Context Dilution in Multi-Hop RAG) — https://arxiv.org/abs/2512.10787
- AdaGATE — https://arxiv.org/abs/2605.05245
- HopRAG — https://arxiv.org/abs/2502.12442
- RouteRAG — https://arxiv.org/abs/2512.09487
- AT-RAG — https://arxiv.org/abs/2410.12886
- CQC-RAG — https://arxiv.org/abs/2606.13438
- LangGraph Plan-and-Execute Agents — https://www.langchain.com/blog/planning-agents
- ReWOO Agent Pattern (docs) — https://agent-patterns.readthedocs.io/en/stable/patterns/rewoo.html
- ReWOO vs. ReAct — https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/
- The 4 Single-Agent Patterns — https://theaiengineer.substack.com/p/the-4-single-agent-patterns
- HippoRAG 2 Overview (Emergent Mind) — https://www.emergentmind.com/topics/hipporag-2
- GraphRAG vs HippoRAG vs PathRAG vs OG-RAG — https://medium.com/graph-praxis/graphrag-vs-hipporag-vs-pathrag-vs-og-rag-choosing-the-right-architecture-for-your-knowledge-graph-a4745e8b125f
- Enhancing HippoRAG with Graph-Based Semantics — https://graphwise.ai/blog/from-retrieval-to-reasoning-enhancing-hipporag-with-graph-based-semantics/
- 12 Advanced RAG Techniques — https://atlan.com/know/advanced-rag-techniques/
