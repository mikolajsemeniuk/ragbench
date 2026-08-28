# Rag
* Setup
* Health check the embedder
* Ingest data
* Measure corpus coverage
* Run benchmark
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
perfect retriever would score 0.726 there and the measured 0.30 is 41% of what
is achievable, not 30% of it.

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
| `adaptive` | one call routes to closedbook, naive or ircot | all 5 | 2-7 |

Every recipe is a plain `go run` line, so a single question set can be run by
copying the one line out of the Makefile and editing it - use a scratch `-dump`
path if you add `-limit` for a smoke test, or the real files get overwritten
with a sample.

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
| naive12 vs neighbour | neighbour puts ~11.7 passages in context |
| naive vs adaptive, ircot vs adaptive | a router is only interesting against the branches it picks between |

`make all` runs the benchmarks, the diagnosis and the comparisons in order.

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
- Navigating Modern LLM Agent Architectures — https://www.wollenlabs.com/blog-posts/navigating-modern-llm-agent-architectures-multi-agents-plan-and-execute-rewoo-tree-of-thoughts-and-react
- HippoRAG 2 Overview (Emergent Mind) — https://www.emergentmind.com/topics/hipporag-2
- GraphRAG vs HippoRAG vs PathRAG vs OG-RAG — https://medium.com/graph-praxis/graphrag-vs-hipporag-vs-pathrag-vs-og-rag-choosing-the-right-architecture-for-your-knowledge-graph-a4745e8b125f
- Enhancing HippoRAG with Graph-Based Semantics — https://graphwise.ai/blog/from-retrieval-to-reasoning-enhancing-hipporag-with-graph-based-semantics/
- 12 Advanced RAG Techniques — https://atlan.com/know/advanced-rag-techniques/
