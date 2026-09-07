# Every experiment in the paper, one target per architecture.
# Each line is a command you can also copy and run by hand.

.PHONY: all ingest ingest-sparse coverage bench closedbook naive-rag \
	naive-rag-ten naive-rag-twelve bm25 hybrid hyde rerank neighbour crag \
	crag-ten ircot adaptive fused cascade cascade-rerank diagnose compare

all: bench diagnose compare coverage

# Approx 3h on the GPU. An interrupted run prints the line to resume from;
# re-run it by hand with -skip N.
ingest:
	go run ./cmd/ingest -input dataset/wiki18_100w.jsonl -provider vllm -embed-url http://localhost:8001 -embed-model bge-base-en-v1.5 -qdrant-url http://localhost:6333 -collection ragbench-wiki18 -batch-size 128 -concurrency 8 -max-tokens 512

# Approx 25 min, no GPU. A separate collection over the same passage ids, so
# the dense one is never rebuilt to add lexical search.
ingest-sparse:
	go run ./cmd/ingest -input dataset/wiki18_100w.jsonl -sparse -collection ragbench-wiki18-bm25 -total 21015324 -batch-size 512 -concurrency 8

# The ceiling Recall cannot exceed, because not every gold article is in the
# corpus. NQ and TriviaQA annotate no gold documents, so they have none.
coverage:
	go run ./cmd/coverage -dataset dataset/musique_dev.jsonl -name CoverageMuSiQue -tex-out paper/coverage-musique.gen.tex
	go run ./cmd/coverage -dataset dataset/hotpotqa_dev.jsonl -name CoverageHotpotQA -tex-out paper/coverage-hotpotqa.gen.tex
	go run ./cmd/coverage -dataset dataset/2wikimultihopqa_dev.jsonl -name CoverageTwoWiki -tex-out paper/coverage-2wiki.gen.tex

# Cheapest first, so a broken stack fails in minutes rather than hours.
bench: closedbook naive-rag naive-rag-ten naive-rag-twelve bm25 hybrid hyde rerank neighbour crag crag-ten ircot adaptive cascade

closedbook:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-musique.jsonl -name ClosedBookMuSiQue -tex-out paper/closedbook-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-hotpotqa.jsonl -name ClosedBookHotpotQA -tex-out paper/closedbook-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-2wiki.jsonl -name ClosedBookTwoWiki -tex-out paper/closedbook-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-nq.jsonl -name ClosedBookNQ -tex-out paper/closedbook-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-triviaqa.jsonl -name ClosedBookTriviaQA -tex-out paper/closedbook-triviaqa.gen.tex

naive-rag:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-musique.jsonl -name NaiveRAGMuSiQue -tex-out paper/naive-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-hotpotqa.jsonl -name NaiveRAGHotpotQA -tex-out paper/naive-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-2wiki.jsonl -name NaiveRAGTwoWiki -tex-out paper/naive-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-nq.jsonl -name NaiveRAGNQ -tex-out paper/naive-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-triviaqa.jsonl -name NaiveRAGTriviaQA -tex-out paper/naive-triviaqa.gen.tex

# Matched-budget control for IRCoT and CRAG.
naive-rag-ten:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 10 -concurrency 48 -dump runs/naive10-musique.jsonl -name NaiveRAGTopTenMuSiQue -tex-out paper/naive10-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 10 -concurrency 48 -dump runs/naive10-hotpotqa.jsonl -name NaiveRAGTopTenHotpotQA -tex-out paper/naive10-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 10 -concurrency 48 -dump runs/naive10-2wiki.jsonl -name NaiveRAGTopTenTwoWiki -tex-out paper/naive10-2wiki.gen.tex

# Matched-budget control for neighbour expansion, which puts ~11.7 passages in
# context. Without it that row measures context size, not chunking.
naive-rag-twelve:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 12 -concurrency 48 -dump runs/naive12-musique.jsonl -name NaiveRAGTopTwelveMuSiQue -tex-out paper/naive12-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 12 -concurrency 48 -dump runs/naive12-hotpotqa.jsonl -name NaiveRAGTopTwelveHotpotQA -tex-out paper/naive12-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 12 -concurrency 48 -dump runs/naive12-2wiki.jsonl -name NaiveRAGTopTwelveTwoWiki -tex-out paper/naive12-2wiki.gen.tex

bm25:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-musique.jsonl -name BMTwentyFiveMuSiQue -tex-out paper/bm25-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-hotpotqa.jsonl -name BMTwentyFiveHotpotQA -tex-out paper/bm25-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-2wiki.jsonl -name BMTwentyFiveTwoWiki -tex-out paper/bm25-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-nq.jsonl -name BMTwentyFiveNQ -tex-out paper/bm25-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-triviaqa.jsonl -name BMTwentyFiveTriviaQA -tex-out paper/bm25-triviaqa.gen.tex

hybrid:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-musique.jsonl -name HybridMuSiQue -tex-out paper/hybrid-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-hotpotqa.jsonl -name HybridHotpotQA -tex-out paper/hybrid-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-2wiki.jsonl -name HybridTwoWiki -tex-out paper/hybrid-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-nq.jsonl -name HybridNQ -tex-out paper/hybrid-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-triviaqa.jsonl -name HybridTriviaQA -tex-out paper/hybrid-triviaqa.gen.tex

hyde:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-musique.jsonl -name HyDEMuSiQue -tex-out paper/hyde-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-hotpotqa.jsonl -name HyDEHotpotQA -tex-out paper/hyde-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-2wiki.jsonl -name HyDETwoWiki -tex-out paper/hyde-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-nq.jsonl -name HyDENQ -tex-out paper/hyde-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-triviaqa.jsonl -name HyDETriviaQA -tex-out paper/hyde-triviaqa.gen.tex

rerank:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-musique.jsonl -name RerankMuSiQue -tex-out paper/rerank-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-hotpotqa.jsonl -name RerankHotpotQA -tex-out paper/rerank-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-2wiki.jsonl -name RerankTwoWiki -tex-out paper/rerank-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-nq.jsonl -name RerankNQ -tex-out paper/rerank-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-triviaqa.jsonl -name RerankTriviaQA -tex-out paper/rerank-triviaqa.gen.tex

# The first run builds dataset/wiki18_100w.titles.gob (~1.5 min) and reuses it.
neighbour:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture neighbour -concurrency 48 -dump runs/neighbour-musique.jsonl -name NeighbourMuSiQue -tex-out paper/neighbour-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture neighbour -concurrency 48 -dump runs/neighbour-hotpotqa.jsonl -name NeighbourHotpotQA -tex-out paper/neighbour-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture neighbour -concurrency 48 -dump runs/neighbour-2wiki.jsonl -name NeighbourTwoWiki -tex-out paper/neighbour-2wiki.gen.tex

crag:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-musique.jsonl -name CRAGMuSiQue -tex-out paper/crag-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-hotpotqa.jsonl -name CRAGHotpotQA -tex-out paper/crag-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-2wiki.jsonl -name CRAGTwoWiki -tex-out paper/crag-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-nq.jsonl -name CRAGNQ -tex-out paper/crag-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-triviaqa.jsonl -name CRAGTriviaQA -tex-out paper/crag-triviaqa.gen.tex

crag-ten:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture crag -top-k 10 -crag-max-passages 10 -concurrency 48 -dump runs/crag10-musique.jsonl -name CRAGTopTenMuSiQue -tex-out paper/crag10-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -top-k 10 -crag-max-passages 10 -concurrency 48 -dump runs/crag10-hotpotqa.jsonl -name CRAGTopTenHotpotQA -tex-out paper/crag10-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -top-k 10 -crag-max-passages 10 -concurrency 48 -dump runs/crag10-2wiki.jsonl -name CRAGTopTenTwoWiki -tex-out paper/crag10-2wiki.gen.tex

ircot:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-musique.jsonl -name IRCoTMuSiQue -tex-out paper/ircot-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-hotpotqa.jsonl -name IRCoTHotpotQA -tex-out paper/ircot-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-2wiki.jsonl -name IRCoTTwoWiki -tex-out paper/ircot-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-nq.jsonl -name IRCoTNQ -tex-out paper/ircot-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-triviaqa.jsonl -name IRCoTTriviaQA -tex-out paper/ircot-triviaqa.gen.tex

adaptive:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-musique.jsonl -name AdaptiveMuSiQue -tex-out paper/adaptive-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-hotpotqa.jsonl -name AdaptiveHotpotQA -tex-out paper/adaptive-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-2wiki.jsonl -name AdaptiveTwoWiki -tex-out paper/adaptive-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-nq.jsonl -name AdaptiveNQ -tex-out paper/adaptive-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-triviaqa.jsonl -name AdaptiveTriviaQA -tex-out paper/adaptive-triviaqa.gen.tex

# ABLATIONS. Not part of `make bench` - the architecture is `cascade`.
#
# Its Exact Match ablation needs no run at all: in the cascade, stage 1 answers
# every question, and every escalated question is one where stage 1 abstained,
# which scores Exact Match 0 by construction. So stage-1 EM is recoverable from
# runs/cascade-*.jsonl by summing EM over the rows whose "stage" is the first
# stage. Run this target only for the metrics that are NOT recoverable that way
# - Recall, MRR and answer-in-context of stage 1 on its own.
#
# The two comparisons that need these runs (Rerank vs Fused, Fused vs Cascade)
# live here rather than in `cascade` or `compare`, so that `make bench` and
# `make all` do not fail on a run they never produce. Run `make cascade` first.
fused:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-musique.jsonl -name FusedMuSiQue -tex-out paper/fused-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-hotpotqa.jsonl -name FusedHotpotQA -tex-out paper/fused-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-2wiki.jsonl -name FusedTwoWiki -tex-out paper/fused-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-nq.jsonl -name FusedNQ -tex-out paper/fused-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-triviaqa.jsonl -name FusedTriviaQA -tex-out paper/fused-triviaqa.gen.tex
	go run ./cmd/compare -a runs/rerank-musique.jsonl -b runs/fused-musique.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedMuSiQue -tex-out paper/cmp-rerank-fused-musique.gen.tex
	go run ./cmd/compare -a runs/fused-musique.jsonl -b runs/cascade-musique.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeMuSiQue -tex-out paper/cmp-fused-cascade-musique.gen.tex
	go run ./cmd/compare -a runs/rerank-hotpotqa.jsonl -b runs/fused-hotpotqa.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedHotpotQA -tex-out paper/cmp-rerank-fused-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/fused-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeHotpotQA -tex-out paper/cmp-fused-cascade-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/rerank-2wiki.jsonl -b runs/fused-2wiki.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedTwoWiki -tex-out paper/cmp-rerank-fused-2wiki.gen.tex
	go run ./cmd/compare -a runs/fused-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeTwoWiki -tex-out paper/cmp-fused-cascade-2wiki.gen.tex
	go run ./cmd/compare -a runs/rerank-nq.jsonl -b runs/fused-nq.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedNQ -tex-out paper/cmp-rerank-fused-nq.gen.tex
	go run ./cmd/compare -a runs/fused-nq.jsonl -b runs/cascade-nq.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeNQ -tex-out paper/cmp-fused-cascade-nq.gen.tex
	go run ./cmd/compare -a runs/rerank-triviaqa.jsonl -b runs/fused-triviaqa.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedTriviaQA -tex-out paper/cmp-rerank-fused-triviaqa.gen.tex
	go run ./cmd/compare -a runs/fused-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeTriviaQA -tex-out paper/cmp-fused-cascade-triviaqa.gen.tex

# The proposed architecture: fused retrieval, escalating only the questions the
# reader itself declined to answer. Its Recall and MRR are NOT comparable with
# a pure retrieval system's - questions that reach the closed-book stage end
# with no passages at all - so compare it on Exact Match.
cascade:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-musique.jsonl -name CascadeMuSiQue -tex-out paper/cascade-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-hotpotqa.jsonl -name CascadeHotpotQA -tex-out paper/cascade-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-2wiki.jsonl -name CascadeTwoWiki -tex-out paper/cascade-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-nq.jsonl -name CascadeNQ -tex-out paper/cascade-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-triviaqa.jsonl -name CascadeTriviaQA -tex-out paper/cascade-triviaqa.gen.tex

# Isolates what the fusion contributes: the same cascade with plain reranking
# as stage 1.
cascade-rerank:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-musique.jsonl -name CascadeRerankMuSiQue -tex-out paper/cascade-rr-musique.gen.tex
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-hotpotqa.jsonl -name CascadeRerankHotpotQA -tex-out paper/cascade-rr-hotpotqa.gen.tex
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-2wiki.jsonl -name CascadeRerankTwoWiki -tex-out paper/cascade-rr-2wiki.gen.tex
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-nq.jsonl -name CascadeRerankNQ -tex-out paper/cascade-rr-nq.gen.tex
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-triviaqa.jsonl -name CascadeRerankTriviaQA -tex-out paper/cascade-rr-triviaqa.gen.tex

# Needs the naive and closedbook runs.
diagnose:
	go run ./cmd/diagnose -dataset dataset/musique_dev.jsonl -run runs/naive-musique.jsonl -closedbook runs/closedbook-musique.jsonl -collection ragbench-wiki18 -top-k 5 -name DiagMuSiQue -tex-out paper/diagnosis-musique.gen.tex
	go run ./cmd/diagnose -dataset dataset/hotpotqa_dev.jsonl -run runs/naive-hotpotqa.jsonl -closedbook runs/closedbook-hotpotqa.jsonl -collection ragbench-wiki18 -top-k 5 -name DiagHotpotQA -tex-out paper/diagnosis-hotpotqa.gen.tex
	go run ./cmd/diagnose -dataset dataset/2wikimultihopqa_dev.jsonl -run runs/naive-2wiki.jsonl -closedbook runs/closedbook-2wiki.jsonl -collection ragbench-wiki18 -top-k 5 -name DiagTwoWiki -tex-out paper/diagnosis-2wiki.gen.tex

# bm25, hybrid, hyde and rerank keep the 5-passage budget, so naive is the
# control. ircot, crag10 and neighbour do not, so they go against naive10/12.
compare:
	go run ./cmd/compare -a runs/closedbook-musique.jsonl -b runs/naive-musique.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveMuSiQue -tex-out paper/cmp-closedbook-naive-musique.gen.tex
	go run ./cmd/compare -a runs/closedbook-hotpotqa.jsonl -b runs/naive-hotpotqa.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveHotpotQA -tex-out paper/cmp-closedbook-naive-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/closedbook-2wiki.jsonl -b runs/naive-2wiki.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveTwoWiki -tex-out paper/cmp-closedbook-naive-2wiki.gen.tex
	go run ./cmd/compare -a runs/closedbook-nq.jsonl -b runs/naive-nq.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveNQ -tex-out paper/cmp-closedbook-naive-nq.gen.tex
	go run ./cmd/compare -a runs/closedbook-triviaqa.jsonl -b runs/naive-triviaqa.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveTriviaQA -tex-out paper/cmp-closedbook-naive-triviaqa.gen.tex
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/bm25-musique.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveMuSiQue -tex-out paper/cmp-naive-bm25-musique.gen.tex
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/bm25-hotpotqa.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveHotpotQA -tex-out paper/cmp-naive-bm25-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/bm25-2wiki.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveTwoWiki -tex-out paper/cmp-naive-bm25-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/hybrid-musique.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridMuSiQue -tex-out paper/cmp-naive-hybrid-musique.gen.tex
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/hybrid-hotpotqa.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridHotpotQA -tex-out paper/cmp-naive-hybrid-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/hybrid-2wiki.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridTwoWiki -tex-out paper/cmp-naive-hybrid-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/hyde-musique.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDEMuSiQue -tex-out paper/cmp-naive-hyde-musique.gen.tex
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/hyde-hotpotqa.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDEHotpotQA -tex-out paper/cmp-naive-hyde-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/hyde-2wiki.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDETwoWiki -tex-out paper/cmp-naive-hyde-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/rerank-musique.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankMuSiQue -tex-out paper/cmp-naive-rerank-musique.gen.tex
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/rerank-hotpotqa.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankHotpotQA -tex-out paper/cmp-naive-rerank-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/rerank-2wiki.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankTwoWiki -tex-out paper/cmp-naive-rerank-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive10-musique.jsonl -b runs/ircot-musique.jsonl -name-a NaiveTen -name-b IRCoT -name NaiveTenVsIRCoTMuSiQue -tex-out paper/cmp-naive10-ircot-musique.gen.tex
	go run ./cmd/compare -a runs/naive10-hotpotqa.jsonl -b runs/ircot-hotpotqa.jsonl -name-a NaiveTen -name-b IRCoT -name NaiveTenVsIRCoTHotpotQA -tex-out paper/cmp-naive10-ircot-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive10-2wiki.jsonl -b runs/ircot-2wiki.jsonl -name-a NaiveTen -name-b IRCoT -name NaiveTenVsIRCoTTwoWiki -tex-out paper/cmp-naive10-ircot-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive10-musique.jsonl -b runs/crag10-musique.jsonl -name-a NaiveTen -name-b CRAGTen -name NaiveTenVsCRAGTenMuSiQue -tex-out paper/cmp-naive10-crag10-musique.gen.tex
	go run ./cmd/compare -a runs/naive10-hotpotqa.jsonl -b runs/crag10-hotpotqa.jsonl -name-a NaiveTen -name-b CRAGTen -name NaiveTenVsCRAGTenHotpotQA -tex-out paper/cmp-naive10-crag10-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive10-2wiki.jsonl -b runs/crag10-2wiki.jsonl -name-a NaiveTen -name-b CRAGTen -name NaiveTenVsCRAGTenTwoWiki -tex-out paper/cmp-naive10-crag10-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive12-musique.jsonl -b runs/neighbour-musique.jsonl -name-a NaiveTwelve -name-b Neighbour -name NaiveTwelveVsNeighbourMuSiQue -tex-out paper/cmp-naive12-neighbour-musique.gen.tex
	go run ./cmd/compare -a runs/naive12-hotpotqa.jsonl -b runs/neighbour-hotpotqa.jsonl -name-a NaiveTwelve -name-b Neighbour -name NaiveTwelveVsNeighbourHotpotQA -tex-out paper/cmp-naive12-neighbour-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive12-2wiki.jsonl -b runs/neighbour-2wiki.jsonl -name-a NaiveTwelve -name-b Neighbour -name NaiveTwelveVsNeighbourTwoWiki -tex-out paper/cmp-naive12-neighbour-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/adaptive-musique.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveMuSiQue -tex-out paper/cmp-naive-adaptive-musique.gen.tex
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/adaptive-hotpotqa.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveHotpotQA -tex-out paper/cmp-naive-adaptive-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/adaptive-2wiki.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveTwoWiki -tex-out paper/cmp-naive-adaptive-2wiki.gen.tex
	go run ./cmd/compare -a runs/ircot-musique.jsonl -b runs/adaptive-musique.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveMuSiQue -tex-out paper/cmp-ircot-adaptive-musique.gen.tex
	go run ./cmd/compare -a runs/ircot-hotpotqa.jsonl -b runs/adaptive-hotpotqa.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveHotpotQA -tex-out paper/cmp-ircot-adaptive-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/ircot-2wiki.jsonl -b runs/adaptive-2wiki.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveTwoWiki -tex-out paper/cmp-ircot-adaptive-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/bm25-nq.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveNQ -tex-out paper/cmp-naive-bm25-nq.gen.tex
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/bm25-triviaqa.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveTriviaQA -tex-out paper/cmp-naive-bm25-triviaqa.gen.tex
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/hybrid-nq.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridNQ -tex-out paper/cmp-naive-hybrid-nq.gen.tex
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/hybrid-triviaqa.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridTriviaQA -tex-out paper/cmp-naive-hybrid-triviaqa.gen.tex
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/hyde-nq.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDENQ -tex-out paper/cmp-naive-hyde-nq.gen.tex
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/hyde-triviaqa.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDETriviaQA -tex-out paper/cmp-naive-hyde-triviaqa.gen.tex
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/rerank-nq.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankNQ -tex-out paper/cmp-naive-rerank-nq.gen.tex
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/rerank-triviaqa.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankTriviaQA -tex-out paper/cmp-naive-rerank-triviaqa.gen.tex
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/adaptive-nq.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveNQ -tex-out paper/cmp-naive-adaptive-nq.gen.tex
	go run ./cmd/compare -a runs/ircot-nq.jsonl -b runs/adaptive-nq.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveNQ -tex-out paper/cmp-ircot-adaptive-nq.gen.tex
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/adaptive-triviaqa.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveTriviaQA -tex-out paper/cmp-naive-adaptive-triviaqa.gen.tex
	go run ./cmd/compare -a runs/ircot-triviaqa.jsonl -b runs/adaptive-triviaqa.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveTriviaQA -tex-out paper/cmp-ircot-adaptive-triviaqa.gen.tex
	go run ./cmd/compare -a runs/rerank-musique.jsonl -b runs/cascade-musique.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeMuSiQue -tex-out paper/cmp-rerank-cascade-musique.gen.tex
	go run ./cmd/compare -a runs/rerank-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeHotpotQA -tex-out paper/cmp-rerank-cascade-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/rerank-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeTwoWiki -tex-out paper/cmp-rerank-cascade-2wiki.gen.tex
	go run ./cmd/compare -a runs/rerank-nq.jsonl -b runs/cascade-nq.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeNQ -tex-out paper/cmp-rerank-cascade-nq.gen.tex
	go run ./cmd/compare -a runs/rerank-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeTriviaQA -tex-out paper/cmp-rerank-cascade-triviaqa.gen.tex
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/cascade-musique.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeMuSiQue -tex-out paper/cmp-naive-cascade-musique.gen.tex
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeHotpotQA -tex-out paper/cmp-naive-cascade-hotpotqa.gen.tex
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeTwoWiki -tex-out paper/cmp-naive-cascade-2wiki.gen.tex
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/cascade-nq.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeNQ -tex-out paper/cmp-naive-cascade-nq.gen.tex
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeTriviaQA -tex-out paper/cmp-naive-cascade-triviaqa.gen.tex
