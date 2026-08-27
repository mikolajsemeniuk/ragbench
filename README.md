# Rag
* Setup
* Health check the embedder
* Ingest data
* Run benchmark
* Check ingested data
* Clean collection
* Dataset
* Groups
* RAG Measurements 
* Generation Measurements
* Bibliography

## Setup

```sh
docker compose up -d qdrant vllm-embed vllm-llm
```

## Health check the embedder

```sh
curl -s http://localhost:8001/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"bge-base-en-v1.5","input":["hello world"],"truncate_prompt_tokens":512}' | jq '{dim:(.data[0].embedding|length), tokens:.usage.prompt_tokens}'
```

## Ingest data

```sh
# Approx 3h
go run ./cmd/ingest -input dataset/wiki18_100w.jsonl -provider vllm -embed-url http://localhost:8001 -embed-model bge-base-en-v1.5 -qdrant-url http://localhost:6333 -collection ragbench-wiki18 -batch-size 128 -concurrency 8 -max-tokens 512
```

## Run benchmark

```sh
# Smoke test, ~1 min
go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -limit 200 -concurrency 8

go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-musique.jsonl -name NaiveRAGMuSiQue -tex-out paper/naive-musique.gen.tex
go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-hotpotqa.jsonl -name NaiveRAGHotpotQA -tex-out paper/naive-hotpotqa.gen.tex
go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-2wiki.jsonl -name NaiveRAGTwoWiki -tex-out paper/naive-2wiki.gen.tex
go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-nq.jsonl -name NaiveRAGNQ -tex-out paper/naive-nq.gen.tex
go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-triviaqa.jsonl -name NaiveRAGTriviaQA -tex-out paper/naive-triviaqa.gen.tex

go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-musique.jsonl -name IRCoTMuSiQue -tex-out paper/ircot-musique.gen.tex
go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-hotpotqa.jsonl -name IRCoTHotpotQA -tex-out paper/ircot-hotpotqa.gen.tex
go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-2wiki.jsonl -name IRCoTTwoWiki -tex-out paper/ircot-2wiki.gen.tex
go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-nq.jsonl -name IRCoTNQ -tex-out paper/ircot-nq.gen.tex
go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-triviaqa.jsonl -name IRCoTTriviaQA -tex-out paper/ircot-triviaqa.gen.tex

go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-musique.jsonl -name CRAGMuSiQue -tex-out paper/crag-musique.gen.tex
go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-hotpotqa.jsonl -name CRAGHotpotQA -tex-out paper/crag-hotpotqa.gen.tex
go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-2wiki.jsonl -name CRAGTwoWiki -tex-out paper/crag-2wiki.gen.tex
go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-nq.jsonl -name CRAGNQ -tex-out paper/crag-nq.gen.tex
go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-triviaqa.jsonl -name CRAGTriviaQA -tex-out paper/crag-triviaqa.gen.tex

go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-musique.jsonl -name ClosedBookMuSiQue -tex-out paper/closedbook-musique.gen.tex
go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-hotpotqa.jsonl -name ClosedBookHotpotQA -tex-out paper/closedbook-hotpotqa.gen.tex
go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-2wiki.jsonl -name ClosedBookTwoWiki -tex-out paper/closedbook-2wiki.gen.tex
go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-nq.jsonl -name ClosedBookNQ -tex-out paper/closedbook-nq.gen.tex
go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-triviaqa.jsonl -name ClosedBookTriviaQA -tex-out paper/closedbook-triviaqa.gen.tex

go run ./cmd/compare -a runs/naive10-musique.jsonl -b runs/ircot-musique.jsonl -name-a Naive10 -name-b IRCoT
go run ./cmd/compare -a runs/naive10-hotpotqa.jsonl -b runs/ircot-hotpotqa.jsonl -name-a Naive10 -name-b IRCoT
go run ./cmd/compare -a runs/naive10-2wiki.jsonl -b runs/ircot-2wiki.jsonl -name-a Naive10 -name-b IRCoT
go run ./cmd/compare -a runs/closedbook-2wiki.jsonl -b runs/naive-2wiki.jsonl -name-a ClosedBook -name-b NaiveRAG
go run ./cmd/compare -a runs/closedbook-triviaqa.jsonl -b runs/naive-triviaqa.jsonl -name-a ClosedBook -name-b NaiveRAG
```

## Check ingested data

```sh
curl -s http://localhost:6333/collections/ragbench-test | jq '.result.points_count'
```

## Clean collection

```sh
curl -s -X DELETE http://localhost:6333/collections/ragbench-test
```

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
  * Recall@K: in how many cases does the model retrieve the correct document. Recall@5 = 0.8 means the model retrieved the correct document in 80% of top 5 retrieved documents.
  * Precision@K: how many of the top K retrieved documents are relevant. Precision@5 = 0.8 means 80% of top 5 retrieved documents are relevant.
  * MRR (Mean Reciprocal Rank): the average of the reciprocal ranks of the first relevant document for each query.

## Generation Measurements
  * Exact Match (EM): does the response match the answer exactly.
  * F1 (token-level): the F1 score at the token level.
  * Faithfulness/groundedness: how well the model's response is grounded in the retrieved documents. For instance, LLM as judge is used to score the response.
  * Answer Relevance: how well the model's response is relevant to the query.
  * Number of tokens: the number of tokens in the model's response.
  * Latency: the time it takes for the model to generate a response.

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
