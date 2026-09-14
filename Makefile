# Every experiment in the paper, one target per architecture.
# Each line is a command you can also copy and run by hand.

.PHONY: all ingest ingest-sparse coverage bench closedbook naive-rag \
	naive-rag-ten naive-rag-twelve bm25 hybrid hyde rerank neighbour crag \
	crag-ten ircot ircot-oneshot adaptive cascade cascade-ablations cascade-tune \
	cascade-logprob reader result cost diagnose compare render pdf

all: bench diagnose compare coverage result cost render

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
	go run ./cmd/coverage -dataset dataset/musique_dev.jsonl -name CoverageMuSiQue -json-out eval/coverage-musique.json
	go run ./cmd/coverage -dataset dataset/hotpotqa_dev.jsonl -name CoverageHotpotQA -json-out eval/coverage-hotpotqa.json
	go run ./cmd/coverage -dataset dataset/2wikimultihopqa_dev.jsonl -name CoverageTwoWiki -json-out eval/coverage-2wiki.json

# Cheapest first, so a broken stack fails in minutes rather than hours.
bench: closedbook naive-rag naive-rag-ten naive-rag-twelve bm25 hybrid hyde rerank neighbour crag crag-ten ircot adaptive cascade

closedbook:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-musique.jsonl -name ClosedBookMuSiQue -json-out eval/closedbook-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-hotpotqa.jsonl -name ClosedBookHotpotQA -json-out eval/closedbook-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-2wiki.jsonl -name ClosedBookTwoWiki -json-out eval/closedbook-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-nq.jsonl -name ClosedBookNQ -json-out eval/closedbook-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture closedbook -concurrency 48 -dump runs/closedbook-triviaqa.jsonl -name ClosedBookTriviaQA -json-out eval/closedbook-triviaqa.json

naive-rag:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-musique.jsonl -name NaiveRAGMuSiQue -json-out eval/naive-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-hotpotqa.jsonl -name NaiveRAGHotpotQA -json-out eval/naive-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-2wiki.jsonl -name NaiveRAGTwoWiki -json-out eval/naive-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-nq.jsonl -name NaiveRAGNQ -json-out eval/naive-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture naive -concurrency 48 -dump runs/naive-triviaqa.jsonl -name NaiveRAGTriviaQA -json-out eval/naive-triviaqa.json

# Matched-budget control for IRCoT and CRAG.
naive-rag-ten:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 10 -concurrency 48 -dump runs/naive10-musique.jsonl -name NaiveRAGTopTenMuSiQue -json-out eval/naive10-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 10 -concurrency 48 -dump runs/naive10-hotpotqa.jsonl -name NaiveRAGTopTenHotpotQA -json-out eval/naive10-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 10 -concurrency 48 -dump runs/naive10-2wiki.jsonl -name NaiveRAGTopTenTwoWiki -json-out eval/naive10-2wiki.json

# Matched-budget control for neighbour expansion, which puts ~11.7 passages in
# context. Without it that row measures context size, not chunking.
naive-rag-twelve:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 12 -concurrency 48 -dump runs/naive12-musique.jsonl -name NaiveRAGTopTwelveMuSiQue -json-out eval/naive12-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 12 -concurrency 48 -dump runs/naive12-hotpotqa.jsonl -name NaiveRAGTopTwelveHotpotQA -json-out eval/naive12-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture naive -top-k 12 -concurrency 48 -dump runs/naive12-2wiki.jsonl -name NaiveRAGTopTwelveTwoWiki -json-out eval/naive12-2wiki.json

bm25:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-musique.jsonl -name BMTwentyFiveMuSiQue -json-out eval/bm25-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-hotpotqa.jsonl -name BMTwentyFiveHotpotQA -json-out eval/bm25-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-2wiki.jsonl -name BMTwentyFiveTwoWiki -json-out eval/bm25-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-nq.jsonl -name BMTwentyFiveNQ -json-out eval/bm25-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -sparse-collection ragbench-wiki18-bm25 -architecture bm25 -concurrency 48 -dump runs/bm25-triviaqa.jsonl -name BMTwentyFiveTriviaQA -json-out eval/bm25-triviaqa.json

hybrid:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-musique.jsonl -name HybridMuSiQue -json-out eval/hybrid-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-hotpotqa.jsonl -name HybridHotpotQA -json-out eval/hybrid-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-2wiki.jsonl -name HybridTwoWiki -json-out eval/hybrid-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-nq.jsonl -name HybridNQ -json-out eval/hybrid-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hybrid -concurrency 48 -dump runs/hybrid-triviaqa.jsonl -name HybridTriviaQA -json-out eval/hybrid-triviaqa.json

hyde:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-musique.jsonl -name HyDEMuSiQue -json-out eval/hyde-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-hotpotqa.jsonl -name HyDEHotpotQA -json-out eval/hyde-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-2wiki.jsonl -name HyDETwoWiki -json-out eval/hyde-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-nq.jsonl -name HyDENQ -json-out eval/hyde-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture hyde -concurrency 48 -dump runs/hyde-triviaqa.jsonl -name HyDETriviaQA -json-out eval/hyde-triviaqa.json

rerank:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-musique.jsonl -name RerankMuSiQue -json-out eval/rerank-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-hotpotqa.jsonl -name RerankHotpotQA -json-out eval/rerank-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-2wiki.jsonl -name RerankTwoWiki -json-out eval/rerank-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-nq.jsonl -name RerankNQ -json-out eval/rerank-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture rerank -concurrency 48 -dump runs/rerank-triviaqa.jsonl -name RerankTriviaQA -json-out eval/rerank-triviaqa.json

# The first run builds dataset/wiki18_100w.titles.gob (~1.5 min) and reuses it.
neighbour:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture neighbour -concurrency 48 -dump runs/neighbour-musique.jsonl -name NeighbourMuSiQue -json-out eval/neighbour-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture neighbour -concurrency 48 -dump runs/neighbour-hotpotqa.jsonl -name NeighbourHotpotQA -json-out eval/neighbour-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture neighbour -concurrency 48 -dump runs/neighbour-2wiki.jsonl -name NeighbourTwoWiki -json-out eval/neighbour-2wiki.json

crag:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-musique.jsonl -name CRAGMuSiQue -json-out eval/crag-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-hotpotqa.jsonl -name CRAGHotpotQA -json-out eval/crag-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-2wiki.jsonl -name CRAGTwoWiki -json-out eval/crag-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-nq.jsonl -name CRAGNQ -json-out eval/crag-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture crag -concurrency 48 -dump runs/crag-triviaqa.jsonl -name CRAGTriviaQA -json-out eval/crag-triviaqa.json

crag-ten:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture crag -top-k 10 -crag-max-passages 10 -concurrency 48 -dump runs/crag10-musique.jsonl -name CRAGTopTenMuSiQue -json-out eval/crag10-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -top-k 10 -crag-max-passages 10 -concurrency 48 -dump runs/crag10-hotpotqa.jsonl -name CRAGTopTenHotpotQA -json-out eval/crag10-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture crag -top-k 10 -crag-max-passages 10 -concurrency 48 -dump runs/crag10-2wiki.jsonl -name CRAGTopTenTwoWiki -json-out eval/crag10-2wiki.json

ircot:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-musique.jsonl -name IRCoTMuSiQue -json-out eval/ircot-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-hotpotqa.jsonl -name IRCoTHotpotQA -json-out eval/ircot-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-2wiki.jsonl -name IRCoTTwoWiki -json-out eval/ircot-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-nq.jsonl -name IRCoTNQ -json-out eval/ircot-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture ircot -concurrency 48 -dump runs/ircot-triviaqa.jsonl -name IRCoTTriviaQA -json-out eval/ircot-triviaqa.json

# The one-shot IRCoT, as a separate row. FlashRAG prompts IRCoT with one worked
# example and reports a far larger gain over standard RAG than the zero-shot
# loop above; this run settles whether the baseline was under-prompted. It is
# compared with the zero-shot run, the matched-budget control and the cascade.
ircot-oneshot:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture ircot -ircot-demo -concurrency 48 -dump runs/ircot1-musique.jsonl -name IRCoTOneShotMuSiQue -json-out eval/ircot1-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -ircot-demo -concurrency 48 -dump runs/ircot1-hotpotqa.jsonl -name IRCoTOneShotHotpotQA -json-out eval/ircot1-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture ircot -ircot-demo -concurrency 48 -dump runs/ircot1-2wiki.jsonl -name IRCoTOneShotTwoWiki -json-out eval/ircot1-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture ircot -ircot-demo -concurrency 48 -dump runs/ircot1-nq.jsonl -name IRCoTOneShotNQ -json-out eval/ircot1-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture ircot -ircot-demo -concurrency 48 -dump runs/ircot1-triviaqa.jsonl -name IRCoTOneShotTriviaQA -json-out eval/ircot1-triviaqa.json
	go run ./cmd/compare -a runs/ircot-musique.jsonl -b runs/ircot1-musique.jsonl -name-a IRCoT -name-b IRCoTOneShot -name IRCoTVsIRCoTOneShotMuSiQue -json-out eval/cmp-ircot-ircot1-musique.json
	go run ./cmd/compare -a runs/ircot-hotpotqa.jsonl -b runs/ircot1-hotpotqa.jsonl -name-a IRCoT -name-b IRCoTOneShot -name IRCoTVsIRCoTOneShotHotpotQA -json-out eval/cmp-ircot-ircot1-hotpotqa.json
	go run ./cmd/compare -a runs/ircot-2wiki.jsonl -b runs/ircot1-2wiki.jsonl -name-a IRCoT -name-b IRCoTOneShot -name IRCoTVsIRCoTOneShotTwoWiki -json-out eval/cmp-ircot-ircot1-2wiki.json
	go run ./cmd/compare -a runs/ircot-nq.jsonl -b runs/ircot1-nq.jsonl -name-a IRCoT -name-b IRCoTOneShot -name IRCoTVsIRCoTOneShotNQ -json-out eval/cmp-ircot-ircot1-nq.json
	go run ./cmd/compare -a runs/ircot-triviaqa.jsonl -b runs/ircot1-triviaqa.jsonl -name-a IRCoT -name-b IRCoTOneShot -name IRCoTVsIRCoTOneShotTriviaQA -json-out eval/cmp-ircot-ircot1-triviaqa.json
	go run ./cmd/compare -a runs/naive10-musique.jsonl -b runs/ircot1-musique.jsonl -name-a NaiveTen -name-b IRCoTOneShot -name NaiveTenVsIRCoTOneShotMuSiQue -json-out eval/cmp-naive10-ircot1-musique.json
	go run ./cmd/compare -a runs/naive10-hotpotqa.jsonl -b runs/ircot1-hotpotqa.jsonl -name-a NaiveTen -name-b IRCoTOneShot -name NaiveTenVsIRCoTOneShotHotpotQA -json-out eval/cmp-naive10-ircot1-hotpotqa.json
	go run ./cmd/compare -a runs/naive10-2wiki.jsonl -b runs/ircot1-2wiki.jsonl -name-a NaiveTen -name-b IRCoTOneShot -name NaiveTenVsIRCoTOneShotTwoWiki -json-out eval/cmp-naive10-ircot1-2wiki.json
	go run ./cmd/compare -a runs/ircot1-musique.jsonl -b runs/cascade-musique.jsonl -name-a IRCoTOneShot -name-b Cascade -name IRCoTOneShotVsCascadeMuSiQue -json-out eval/cmp-ircot1-cascade-musique.json
	go run ./cmd/compare -a runs/ircot1-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a IRCoTOneShot -name-b Cascade -name IRCoTOneShotVsCascadeHotpotQA -json-out eval/cmp-ircot1-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/ircot1-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a IRCoTOneShot -name-b Cascade -name IRCoTOneShotVsCascadeTwoWiki -json-out eval/cmp-ircot1-cascade-2wiki.json
	go run ./cmd/compare -a runs/ircot1-nq.jsonl -b runs/cascade-nq.jsonl -name-a IRCoTOneShot -name-b Cascade -name IRCoTOneShotVsCascadeNQ -json-out eval/cmp-ircot1-cascade-nq.json
	go run ./cmd/compare -a runs/ircot1-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a IRCoTOneShot -name-b Cascade -name IRCoTOneShotVsCascadeTriviaQA -json-out eval/cmp-ircot1-cascade-triviaqa.json

adaptive:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-musique.jsonl -name AdaptiveMuSiQue -json-out eval/adaptive-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-hotpotqa.jsonl -name AdaptiveHotpotQA -json-out eval/adaptive-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-2wiki.jsonl -name AdaptiveTwoWiki -json-out eval/adaptive-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-nq.jsonl -name AdaptiveNQ -json-out eval/adaptive-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -architecture adaptive -concurrency 48 -dump runs/adaptive-triviaqa.jsonl -name AdaptiveTriviaQA -json-out eval/adaptive-triviaqa.json

# The proposed architecture: fused retrieval, escalating only the questions the
# reader itself declined to answer. Its Recall and MRR are NOT comparable with
# a pure retrieval system's - questions that reach the closed-book stage end
# with no passages at all - so compare it on Exact Match.
cascade:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-musique.jsonl -name CascadeMuSiQue -json-out eval/cascade-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-hotpotqa.jsonl -name CascadeHotpotQA -json-out eval/cascade-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-2wiki.jsonl -name CascadeTwoWiki -json-out eval/cascade-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-nq.jsonl -name CascadeNQ -json-out eval/cascade-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -concurrency 48 -dump runs/cascade-triviaqa.jsonl -name CascadeTriviaQA -json-out eval/cascade-triviaqa.json

# ABLATIONS of the one architecture above. Not part of `make bench`, because
# none of these is a competing method; every line is the same cascade loop with
# one element switched, selected by -cascade-stages. Run `make cascade` first:
# the comparisons at the end pair each variant with it.
#
#   fused           stage 1 alone, no escalation   -> what the escalation adds
#   cascade-rr      rerank as stage 1              -> what the fusion adds
#   cascade-naive   naive as stage 1               -> the cheapest possible cascade
#   cascade-hybrid  hybrid as stage 1              -> fusion without the cross-encoder
#
# Stage-1 Exact Match needs no run at all: every escalated question is one
# where stage 1 abstained, which scores Exact Match 0 by construction, so it is
# the sum of EM over the rows of runs/cascade-*.jsonl whose "stage" is "fused".
# The fused run exists for the metrics that are NOT recoverable that way -
# stage 1's own Recall, MRR and answer-in-context.
cascade-ablations:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-musique.jsonl -name FusedMuSiQue -json-out eval/fused-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-hotpotqa.jsonl -name FusedHotpotQA -json-out eval/fused-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-2wiki.jsonl -name FusedTwoWiki -json-out eval/fused-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-nq.jsonl -name FusedNQ -json-out eval/fused-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -concurrency 48 -dump runs/fused-triviaqa.jsonl -name FusedTriviaQA -json-out eval/fused-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-musique.jsonl -name CascadeRerankMuSiQue -json-out eval/cascade-rr-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-hotpotqa.jsonl -name CascadeRerankHotpotQA -json-out eval/cascade-rr-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-2wiki.jsonl -name CascadeRerankTwoWiki -json-out eval/cascade-rr-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-nq.jsonl -name CascadeRerankNQ -json-out eval/cascade-rr-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages rerank,hyde,closedbook -concurrency 48 -dump runs/cascade-rr-triviaqa.jsonl -name CascadeRerankTriviaQA -json-out eval/cascade-rr-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages naive,hyde,closedbook -concurrency 48 -dump runs/cascade-naive-musique.jsonl -name CascadeNaiveMuSiQue -json-out eval/cascade-naive-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages naive,hyde,closedbook -concurrency 48 -dump runs/cascade-naive-hotpotqa.jsonl -name CascadeNaiveHotpotQA -json-out eval/cascade-naive-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages naive,hyde,closedbook -concurrency 48 -dump runs/cascade-naive-2wiki.jsonl -name CascadeNaiveTwoWiki -json-out eval/cascade-naive-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages naive,hyde,closedbook -concurrency 48 -dump runs/cascade-naive-nq.jsonl -name CascadeNaiveNQ -json-out eval/cascade-naive-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages naive,hyde,closedbook -concurrency 48 -dump runs/cascade-naive-triviaqa.jsonl -name CascadeNaiveTriviaQA -json-out eval/cascade-naive-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages hybrid,hyde,closedbook -concurrency 48 -dump runs/cascade-hybrid-musique.jsonl -name CascadeHybridMuSiQue -json-out eval/cascade-hybrid-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages hybrid,hyde,closedbook -concurrency 48 -dump runs/cascade-hybrid-hotpotqa.jsonl -name CascadeHybridHotpotQA -json-out eval/cascade-hybrid-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages hybrid,hyde,closedbook -concurrency 48 -dump runs/cascade-hybrid-2wiki.jsonl -name CascadeHybridTwoWiki -json-out eval/cascade-hybrid-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages hybrid,hyde,closedbook -concurrency 48 -dump runs/cascade-hybrid-nq.jsonl -name CascadeHybridNQ -json-out eval/cascade-hybrid-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-stages hybrid,hyde,closedbook -concurrency 48 -dump runs/cascade-hybrid-triviaqa.jsonl -name CascadeHybridTriviaQA -json-out eval/cascade-hybrid-triviaqa.json
	go run ./cmd/compare -a runs/rerank-musique.jsonl -b runs/fused-musique.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedMuSiQue -json-out eval/cmp-rerank-fused-musique.json
	go run ./cmd/compare -a runs/rerank-hotpotqa.jsonl -b runs/fused-hotpotqa.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedHotpotQA -json-out eval/cmp-rerank-fused-hotpotqa.json
	go run ./cmd/compare -a runs/rerank-2wiki.jsonl -b runs/fused-2wiki.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedTwoWiki -json-out eval/cmp-rerank-fused-2wiki.json
	go run ./cmd/compare -a runs/rerank-nq.jsonl -b runs/fused-nq.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedNQ -json-out eval/cmp-rerank-fused-nq.json
	go run ./cmd/compare -a runs/rerank-triviaqa.jsonl -b runs/fused-triviaqa.jsonl -name-a Rerank -name-b Fused -name RerankVsFusedTriviaQA -json-out eval/cmp-rerank-fused-triviaqa.json
	go run ./cmd/compare -a runs/fused-musique.jsonl -b runs/cascade-musique.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeMuSiQue -json-out eval/cmp-fused-cascade-musique.json
	go run ./cmd/compare -a runs/fused-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeHotpotQA -json-out eval/cmp-fused-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/fused-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeTwoWiki -json-out eval/cmp-fused-cascade-2wiki.json
	go run ./cmd/compare -a runs/fused-nq.jsonl -b runs/cascade-nq.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeNQ -json-out eval/cmp-fused-cascade-nq.json
	go run ./cmd/compare -a runs/fused-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a Fused -name-b Cascade -name FusedVsCascadeTriviaQA -json-out eval/cmp-fused-cascade-triviaqa.json
	go run ./cmd/compare -a runs/cascade-rr-musique.jsonl -b runs/cascade-musique.jsonl -name-a CascadeRerank -name-b Cascade -name CascadeRerankVsCascadeMuSiQue -json-out eval/cmp-cascade-rr-cascade-musique.json
	go run ./cmd/compare -a runs/cascade-rr-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a CascadeRerank -name-b Cascade -name CascadeRerankVsCascadeHotpotQA -json-out eval/cmp-cascade-rr-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/cascade-rr-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a CascadeRerank -name-b Cascade -name CascadeRerankVsCascadeTwoWiki -json-out eval/cmp-cascade-rr-cascade-2wiki.json
	go run ./cmd/compare -a runs/cascade-rr-nq.jsonl -b runs/cascade-nq.jsonl -name-a CascadeRerank -name-b Cascade -name CascadeRerankVsCascadeNQ -json-out eval/cmp-cascade-rr-cascade-nq.json
	go run ./cmd/compare -a runs/cascade-rr-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a CascadeRerank -name-b Cascade -name CascadeRerankVsCascadeTriviaQA -json-out eval/cmp-cascade-rr-cascade-triviaqa.json
	go run ./cmd/compare -a runs/cascade-naive-musique.jsonl -b runs/cascade-musique.jsonl -name-a CascadeNaive -name-b Cascade -name CascadeNaiveVsCascadeMuSiQue -json-out eval/cmp-cascade-naive-cascade-musique.json
	go run ./cmd/compare -a runs/cascade-naive-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a CascadeNaive -name-b Cascade -name CascadeNaiveVsCascadeHotpotQA -json-out eval/cmp-cascade-naive-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/cascade-naive-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a CascadeNaive -name-b Cascade -name CascadeNaiveVsCascadeTwoWiki -json-out eval/cmp-cascade-naive-cascade-2wiki.json
	go run ./cmd/compare -a runs/cascade-naive-nq.jsonl -b runs/cascade-nq.jsonl -name-a CascadeNaive -name-b Cascade -name CascadeNaiveVsCascadeNQ -json-out eval/cmp-cascade-naive-cascade-nq.json
	go run ./cmd/compare -a runs/cascade-naive-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a CascadeNaive -name-b Cascade -name CascadeNaiveVsCascadeTriviaQA -json-out eval/cmp-cascade-naive-cascade-triviaqa.json
	go run ./cmd/compare -a runs/cascade-hybrid-musique.jsonl -b runs/cascade-musique.jsonl -name-a CascadeHybrid -name-b Cascade -name CascadeHybridVsCascadeMuSiQue -json-out eval/cmp-cascade-hybrid-cascade-musique.json
	go run ./cmd/compare -a runs/cascade-hybrid-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a CascadeHybrid -name-b Cascade -name CascadeHybridVsCascadeHotpotQA -json-out eval/cmp-cascade-hybrid-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/cascade-hybrid-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a CascadeHybrid -name-b Cascade -name CascadeHybridVsCascadeTwoWiki -json-out eval/cmp-cascade-hybrid-cascade-2wiki.json
	go run ./cmd/compare -a runs/cascade-hybrid-nq.jsonl -b runs/cascade-nq.jsonl -name-a CascadeHybrid -name-b Cascade -name CascadeHybridVsCascadeNQ -json-out eval/cmp-cascade-hybrid-cascade-nq.json
	go run ./cmd/compare -a runs/cascade-hybrid-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a CascadeHybrid -name-b Cascade -name CascadeHybridVsCascadeTriviaQA -json-out eval/cmp-cascade-hybrid-cascade-triviaqa.json

# CONFIDENCE TRIGGER. The cascade can also escalate an answer whose mean token
# log-probability is below a threshold - the free signal for a wrong answer
# given without hesitation, which the abstention trigger cannot see. The
# threshold is chosen on 1000 sampled questions from each TRAIN split, never on
# the test sets the tables report, by composing the three stages offline
# (cmd/sweep). Same -limit and -seed on all three runs, so they see the same
# questions. Approx 1h.
cascade-tune:
	go run ./cmd/bench -dataset dataset/musique_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-fused-musique.jsonl -name TuneFusedMuSiQue
	go run ./cmd/bench -dataset dataset/hotpotqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-fused-hotpotqa.jsonl -name TuneFusedHotpotQA
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-fused-2wiki.jsonl -name TuneFusedTwoWiki
	go run ./cmd/bench -dataset dataset/naturalquestions_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-fused-nq.jsonl -name TuneFusedNQ
	go run ./cmd/bench -dataset dataset/triviaqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-fused-triviaqa.jsonl -name TuneFusedTriviaQA
	go run ./cmd/bench -dataset dataset/musique_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hyde -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-hyde-musique.jsonl -name TuneHydeMuSiQue
	go run ./cmd/bench -dataset dataset/hotpotqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hyde -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-hyde-hotpotqa.jsonl -name TuneHydeHotpotQA
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hyde -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-hyde-2wiki.jsonl -name TuneHydeTwoWiki
	go run ./cmd/bench -dataset dataset/naturalquestions_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hyde -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-hyde-nq.jsonl -name TuneHydeNQ
	go run ./cmd/bench -dataset dataset/triviaqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture hyde -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-hyde-triviaqa.jsonl -name TuneHydeTriviaQA
	go run ./cmd/bench -dataset dataset/musique_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-closedbook-musique.jsonl -name TuneClosedbookMuSiQue
	go run ./cmd/bench -dataset dataset/hotpotqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-closedbook-hotpotqa.jsonl -name TuneClosedbookHotpotQA
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-closedbook-2wiki.jsonl -name TuneClosedbookTwoWiki
	go run ./cmd/bench -dataset dataset/naturalquestions_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-closedbook-nq.jsonl -name TuneClosedbookNQ
	go run ./cmd/bench -dataset dataset/triviaqa_train.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook -limit 1000 -seed 42 -concurrency 48 -dump runs/tune-closedbook-triviaqa.jsonl -name TuneClosedbookTriviaQA
	go run ./cmd/sweep -stage1 runs/tune-fused-musique.jsonl,runs/tune-fused-hotpotqa.jsonl,runs/tune-fused-2wiki.jsonl,runs/tune-fused-nq.jsonl,runs/tune-fused-triviaqa.jsonl -stage2 runs/tune-hyde-musique.jsonl,runs/tune-hyde-hotpotqa.jsonl,runs/tune-hyde-2wiki.jsonl,runs/tune-hyde-nq.jsonl,runs/tune-hyde-triviaqa.jsonl -last runs/tune-closedbook-musique.jsonl,runs/tune-closedbook-hotpotqa.jsonl,runs/tune-closedbook-2wiki.jsonl,runs/tune-closedbook-nq.jsonl,runs/tune-closedbook-triviaqa.jsonl -name Sweep -json-out eval/sweep.json

# The test-set run at the chosen threshold, as its own row next to the
# abstention-only cascade. Usage: make cascade-logprob TAU=-0.5
cascade-logprob:
	@test -n "$(TAU)" || (echo "set the threshold: make cascade-logprob TAU=<value from make cascade-tune>"; exit 1)
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-min-logprob $(TAU) -concurrency 48 -dump runs/cascade-lp-musique.jsonl -name CascadeLogprobMuSiQue -json-out eval/cascade-lp-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-min-logprob $(TAU) -concurrency 48 -dump runs/cascade-lp-hotpotqa.jsonl -name CascadeLogprobHotpotQA -json-out eval/cascade-lp-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-min-logprob $(TAU) -concurrency 48 -dump runs/cascade-lp-2wiki.jsonl -name CascadeLogprobTwoWiki -json-out eval/cascade-lp-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-min-logprob $(TAU) -concurrency 48 -dump runs/cascade-lp-nq.jsonl -name CascadeLogprobNQ -json-out eval/cascade-lp-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade -cascade-min-logprob $(TAU) -concurrency 48 -dump runs/cascade-lp-triviaqa.jsonl -name CascadeLogprobTriviaQA -json-out eval/cascade-lp-triviaqa.json
	go run ./cmd/compare -a runs/cascade-musique.jsonl -b runs/cascade-lp-musique.jsonl -name-a Cascade -name-b CascadeLogprob -name CascadeVsCascadeLogprobMuSiQue -json-out eval/cmp-cascade-cascade-lp-musique.json
	go run ./cmd/compare -a runs/cascade-hotpotqa.jsonl -b runs/cascade-lp-hotpotqa.jsonl -name-a Cascade -name-b CascadeLogprob -name CascadeVsCascadeLogprobHotpotQA -json-out eval/cmp-cascade-cascade-lp-hotpotqa.json
	go run ./cmd/compare -a runs/cascade-2wiki.jsonl -b runs/cascade-lp-2wiki.jsonl -name-a Cascade -name-b CascadeLogprob -name CascadeVsCascadeLogprobTwoWiki -json-out eval/cmp-cascade-cascade-lp-2wiki.json
	go run ./cmd/compare -a runs/cascade-nq.jsonl -b runs/cascade-lp-nq.jsonl -name-a Cascade -name-b CascadeLogprob -name CascadeVsCascadeLogprobNQ -json-out eval/cmp-cascade-cascade-lp-nq.json
	go run ./cmd/compare -a runs/cascade-triviaqa.jsonl -b runs/cascade-lp-triviaqa.jsonl -name-a Cascade -name-b CascadeLogprob -name CascadeVsCascadeLogprobTriviaQA -json-out eval/cmp-cascade-cascade-lp-triviaqa.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/cascade-lp-musique.jsonl -name-a NaiveRAG -name-b CascadeLogprob -name NaiveRAGVsCascadeLogprobMuSiQue -json-out eval/cmp-naive-cascade-lp-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/cascade-lp-hotpotqa.jsonl -name-a NaiveRAG -name-b CascadeLogprob -name NaiveRAGVsCascadeLogprobHotpotQA -json-out eval/cmp-naive-cascade-lp-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/cascade-lp-2wiki.jsonl -name-a NaiveRAG -name-b CascadeLogprob -name NaiveRAGVsCascadeLogprobTwoWiki -json-out eval/cmp-naive-cascade-lp-2wiki.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/cascade-lp-nq.jsonl -name-a NaiveRAG -name-b CascadeLogprob -name NaiveRAGVsCascadeLogprobNQ -json-out eval/cmp-naive-cascade-lp-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/cascade-lp-triviaqa.jsonl -name-a NaiveRAG -name-b CascadeLogprob -name NaiveRAGVsCascadeLogprobTriviaQA -json-out eval/cmp-naive-cascade-lp-triviaqa.json
	go run ./cmd/compare -a runs/rerank-musique.jsonl -b runs/cascade-lp-musique.jsonl -name-a Rerank -name-b CascadeLogprob -name RerankVsCascadeLogprobMuSiQue -json-out eval/cmp-rerank-cascade-lp-musique.json
	go run ./cmd/compare -a runs/rerank-hotpotqa.jsonl -b runs/cascade-lp-hotpotqa.jsonl -name-a Rerank -name-b CascadeLogprob -name RerankVsCascadeLogprobHotpotQA -json-out eval/cmp-rerank-cascade-lp-hotpotqa.json
	go run ./cmd/compare -a runs/rerank-2wiki.jsonl -b runs/cascade-lp-2wiki.jsonl -name-a Rerank -name-b CascadeLogprob -name RerankVsCascadeLogprobTwoWiki -json-out eval/cmp-rerank-cascade-lp-2wiki.json
	go run ./cmd/compare -a runs/rerank-nq.jsonl -b runs/cascade-lp-nq.jsonl -name-a Rerank -name-b CascadeLogprob -name RerankVsCascadeLogprobNQ -json-out eval/cmp-rerank-cascade-lp-nq.json
	go run ./cmd/compare -a runs/rerank-triviaqa.jsonl -b runs/cascade-lp-triviaqa.jsonl -name-a Rerank -name-b CascadeLogprob -name RerankVsCascadeLogprobTriviaQA -json-out eval/cmp-rerank-cascade-lp-triviaqa.json
	go run ./cmd/result -proposed cascade-lp -name ResultLogprob -json-out eval/result-lp.json

# The summary table: the proposed architecture against every baseline on every
# set, each cell a paired Exact Match difference marked win / loss / tie, with
# Holm over the whole table. Reads runs/, so it comes after bench.
result:
	go run ./cmd/result -json-out eval/result.json

# The cost table: generation calls, prompt and completion tokens and passages
# per question for every architecture, from the dumps. Tokens rather than
# seconds, because a latency at -concurrency 48 measures the queue.
cost:
	go run ./cmd/cost -json-out eval/cost.json

# Renders every eval/*.json aggregate into paper/*.gen.tex. Cheap
# (milliseconds), so it runs before every pdf build; the fragments are a
# cache, the JSON documents in eval/ are the source of truth.
render:
	go run ./cmd/render

# The article itself. Four passes on purpose: bibtex resolves the citations,
# and the two closing pdflatex runs settle the cross-references and the
# bibliography numbers. Needs a LaTeX toolchain (on macOS: MacTeX, which puts
# pdflatex in /Library/TeX/texbin - open a new terminal after installing).
pdf: render
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex

# A SECOND READER. Everything the cascade claims rests on how the reader
# behaves with irrelevant passages, so the rows the claims stand on are
# repeated with another generator: closed-book, naive, rerank, fused and the
# cascade. Same index, same questions; only the generator changes. The runs
# get a tag in their file names so nothing of the main table is overwritten.
#
#   docker compose stop vllm-llm && docker compose up -d vllm-llm-llama
#   make reader
#
# Another model: make reader READER_MODEL=... READER_URL=... READER_TAG=... READER_NAME=...
# READER_NAME is the CamelCase infix of the generated LaTeX commands (letters only).
READER_MODEL ?= llama-3.1-8b-instruct
READER_URL   ?= http://localhost:8003
READER_TAG   ?= llama
READER_NAME  ?= Llama
READER_FLAGS  = -llm-model $(READER_MODEL) -provider-url $(READER_URL)
reader:
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook $(READER_FLAGS) -concurrency 48 -dump runs/closedbook-$(READER_TAG)-musique.jsonl -name ClosedBook$(READER_NAME)MuSiQue -json-out eval/closedbook-$(READER_TAG)-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook $(READER_FLAGS) -concurrency 48 -dump runs/closedbook-$(READER_TAG)-hotpotqa.jsonl -name ClosedBook$(READER_NAME)HotpotQA -json-out eval/closedbook-$(READER_TAG)-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook $(READER_FLAGS) -concurrency 48 -dump runs/closedbook-$(READER_TAG)-2wiki.jsonl -name ClosedBook$(READER_NAME)TwoWiki -json-out eval/closedbook-$(READER_TAG)-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook $(READER_FLAGS) -concurrency 48 -dump runs/closedbook-$(READER_TAG)-nq.jsonl -name ClosedBook$(READER_NAME)NQ -json-out eval/closedbook-$(READER_TAG)-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture closedbook $(READER_FLAGS) -concurrency 48 -dump runs/closedbook-$(READER_TAG)-triviaqa.jsonl -name ClosedBook$(READER_NAME)TriviaQA -json-out eval/closedbook-$(READER_TAG)-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture naive $(READER_FLAGS) -concurrency 48 -dump runs/naive-$(READER_TAG)-musique.jsonl -name NaiveRAG$(READER_NAME)MuSiQue -json-out eval/naive-$(READER_TAG)-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture naive $(READER_FLAGS) -concurrency 48 -dump runs/naive-$(READER_TAG)-hotpotqa.jsonl -name NaiveRAG$(READER_NAME)HotpotQA -json-out eval/naive-$(READER_TAG)-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture naive $(READER_FLAGS) -concurrency 48 -dump runs/naive-$(READER_TAG)-2wiki.jsonl -name NaiveRAG$(READER_NAME)TwoWiki -json-out eval/naive-$(READER_TAG)-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture naive $(READER_FLAGS) -concurrency 48 -dump runs/naive-$(READER_TAG)-nq.jsonl -name NaiveRAG$(READER_NAME)NQ -json-out eval/naive-$(READER_TAG)-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture naive $(READER_FLAGS) -concurrency 48 -dump runs/naive-$(READER_TAG)-triviaqa.jsonl -name NaiveRAG$(READER_NAME)TriviaQA -json-out eval/naive-$(READER_TAG)-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture rerank $(READER_FLAGS) -concurrency 48 -dump runs/rerank-$(READER_TAG)-musique.jsonl -name Rerank$(READER_NAME)MuSiQue -json-out eval/rerank-$(READER_TAG)-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture rerank $(READER_FLAGS) -concurrency 48 -dump runs/rerank-$(READER_TAG)-hotpotqa.jsonl -name Rerank$(READER_NAME)HotpotQA -json-out eval/rerank-$(READER_TAG)-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture rerank $(READER_FLAGS) -concurrency 48 -dump runs/rerank-$(READER_TAG)-2wiki.jsonl -name Rerank$(READER_NAME)TwoWiki -json-out eval/rerank-$(READER_TAG)-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture rerank $(READER_FLAGS) -concurrency 48 -dump runs/rerank-$(READER_TAG)-nq.jsonl -name Rerank$(READER_NAME)NQ -json-out eval/rerank-$(READER_TAG)-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture rerank $(READER_FLAGS) -concurrency 48 -dump runs/rerank-$(READER_TAG)-triviaqa.jsonl -name Rerank$(READER_NAME)TriviaQA -json-out eval/rerank-$(READER_TAG)-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused $(READER_FLAGS) -concurrency 48 -dump runs/fused-$(READER_TAG)-musique.jsonl -name Fused$(READER_NAME)MuSiQue -json-out eval/fused-$(READER_TAG)-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused $(READER_FLAGS) -concurrency 48 -dump runs/fused-$(READER_TAG)-hotpotqa.jsonl -name Fused$(READER_NAME)HotpotQA -json-out eval/fused-$(READER_TAG)-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused $(READER_FLAGS) -concurrency 48 -dump runs/fused-$(READER_TAG)-2wiki.jsonl -name Fused$(READER_NAME)TwoWiki -json-out eval/fused-$(READER_TAG)-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused $(READER_FLAGS) -concurrency 48 -dump runs/fused-$(READER_TAG)-nq.jsonl -name Fused$(READER_NAME)NQ -json-out eval/fused-$(READER_TAG)-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture fused $(READER_FLAGS) -concurrency 48 -dump runs/fused-$(READER_TAG)-triviaqa.jsonl -name Fused$(READER_NAME)TriviaQA -json-out eval/fused-$(READER_TAG)-triviaqa.json
	go run ./cmd/bench -dataset dataset/musique_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade $(READER_FLAGS) -concurrency 48 -dump runs/cascade-$(READER_TAG)-musique.jsonl -name Cascade$(READER_NAME)MuSiQue -json-out eval/cascade-$(READER_TAG)-musique.json
	go run ./cmd/bench -dataset dataset/hotpotqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade $(READER_FLAGS) -concurrency 48 -dump runs/cascade-$(READER_TAG)-hotpotqa.jsonl -name Cascade$(READER_NAME)HotpotQA -json-out eval/cascade-$(READER_TAG)-hotpotqa.json
	go run ./cmd/bench -dataset dataset/2wikimultihopqa_dev.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade $(READER_FLAGS) -concurrency 48 -dump runs/cascade-$(READER_TAG)-2wiki.jsonl -name Cascade$(READER_NAME)TwoWiki -json-out eval/cascade-$(READER_TAG)-2wiki.json
	go run ./cmd/bench -dataset dataset/naturalquestions_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade $(READER_FLAGS) -concurrency 48 -dump runs/cascade-$(READER_TAG)-nq.jsonl -name Cascade$(READER_NAME)NQ -json-out eval/cascade-$(READER_TAG)-nq.json
	go run ./cmd/bench -dataset dataset/triviaqa_test.jsonl -collection ragbench-wiki18 -sparse-collection ragbench-wiki18-bm25 -architecture cascade $(READER_FLAGS) -concurrency 48 -dump runs/cascade-$(READER_TAG)-triviaqa.jsonl -name Cascade$(READER_NAME)TriviaQA -json-out eval/cascade-$(READER_TAG)-triviaqa.json
	go run ./cmd/compare -a runs/closedbook-$(READER_TAG)-musique.jsonl -b runs/naive-$(READER_TAG)-musique.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveRAG$(READER_NAME)MuSiQue -json-out eval/cmp-closedbook-naive-$(READER_TAG)-musique.json
	go run ./cmd/compare -a runs/closedbook-$(READER_TAG)-hotpotqa.jsonl -b runs/naive-$(READER_TAG)-hotpotqa.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveRAG$(READER_NAME)HotpotQA -json-out eval/cmp-closedbook-naive-$(READER_TAG)-hotpotqa.json
	go run ./cmd/compare -a runs/closedbook-$(READER_TAG)-2wiki.jsonl -b runs/naive-$(READER_TAG)-2wiki.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveRAG$(READER_NAME)TwoWiki -json-out eval/cmp-closedbook-naive-$(READER_TAG)-2wiki.json
	go run ./cmd/compare -a runs/closedbook-$(READER_TAG)-nq.jsonl -b runs/naive-$(READER_TAG)-nq.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveRAG$(READER_NAME)NQ -json-out eval/cmp-closedbook-naive-$(READER_TAG)-nq.json
	go run ./cmd/compare -a runs/closedbook-$(READER_TAG)-triviaqa.jsonl -b runs/naive-$(READER_TAG)-triviaqa.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveRAG$(READER_NAME)TriviaQA -json-out eval/cmp-closedbook-naive-$(READER_TAG)-triviaqa.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-musique.jsonl -b runs/rerank-$(READER_TAG)-musique.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveRAGVsRerank$(READER_NAME)MuSiQue -json-out eval/cmp-naive-rerank-$(READER_TAG)-musique.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-hotpotqa.jsonl -b runs/rerank-$(READER_TAG)-hotpotqa.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveRAGVsRerank$(READER_NAME)HotpotQA -json-out eval/cmp-naive-rerank-$(READER_TAG)-hotpotqa.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-2wiki.jsonl -b runs/rerank-$(READER_TAG)-2wiki.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveRAGVsRerank$(READER_NAME)TwoWiki -json-out eval/cmp-naive-rerank-$(READER_TAG)-2wiki.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-nq.jsonl -b runs/rerank-$(READER_TAG)-nq.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveRAGVsRerank$(READER_NAME)NQ -json-out eval/cmp-naive-rerank-$(READER_TAG)-nq.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-triviaqa.jsonl -b runs/rerank-$(READER_TAG)-triviaqa.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveRAGVsRerank$(READER_NAME)TriviaQA -json-out eval/cmp-naive-rerank-$(READER_TAG)-triviaqa.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-musique.jsonl -b runs/fused-$(READER_TAG)-musique.jsonl -name-a Rerank -name-b Fused -name RerankVsFused$(READER_NAME)MuSiQue -json-out eval/cmp-rerank-fused-$(READER_TAG)-musique.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-hotpotqa.jsonl -b runs/fused-$(READER_TAG)-hotpotqa.jsonl -name-a Rerank -name-b Fused -name RerankVsFused$(READER_NAME)HotpotQA -json-out eval/cmp-rerank-fused-$(READER_TAG)-hotpotqa.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-2wiki.jsonl -b runs/fused-$(READER_TAG)-2wiki.jsonl -name-a Rerank -name-b Fused -name RerankVsFused$(READER_NAME)TwoWiki -json-out eval/cmp-rerank-fused-$(READER_TAG)-2wiki.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-nq.jsonl -b runs/fused-$(READER_TAG)-nq.jsonl -name-a Rerank -name-b Fused -name RerankVsFused$(READER_NAME)NQ -json-out eval/cmp-rerank-fused-$(READER_TAG)-nq.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-triviaqa.jsonl -b runs/fused-$(READER_TAG)-triviaqa.jsonl -name-a Rerank -name-b Fused -name RerankVsFused$(READER_NAME)TriviaQA -json-out eval/cmp-rerank-fused-$(READER_TAG)-triviaqa.json
	go run ./cmd/compare -a runs/fused-$(READER_TAG)-musique.jsonl -b runs/cascade-$(READER_TAG)-musique.jsonl -name-a Fused -name-b Cascade -name FusedVsCascade$(READER_NAME)MuSiQue -json-out eval/cmp-fused-cascade-$(READER_TAG)-musique.json
	go run ./cmd/compare -a runs/fused-$(READER_TAG)-hotpotqa.jsonl -b runs/cascade-$(READER_TAG)-hotpotqa.jsonl -name-a Fused -name-b Cascade -name FusedVsCascade$(READER_NAME)HotpotQA -json-out eval/cmp-fused-cascade-$(READER_TAG)-hotpotqa.json
	go run ./cmd/compare -a runs/fused-$(READER_TAG)-2wiki.jsonl -b runs/cascade-$(READER_TAG)-2wiki.jsonl -name-a Fused -name-b Cascade -name FusedVsCascade$(READER_NAME)TwoWiki -json-out eval/cmp-fused-cascade-$(READER_TAG)-2wiki.json
	go run ./cmd/compare -a runs/fused-$(READER_TAG)-nq.jsonl -b runs/cascade-$(READER_TAG)-nq.jsonl -name-a Fused -name-b Cascade -name FusedVsCascade$(READER_NAME)NQ -json-out eval/cmp-fused-cascade-$(READER_TAG)-nq.json
	go run ./cmd/compare -a runs/fused-$(READER_TAG)-triviaqa.jsonl -b runs/cascade-$(READER_TAG)-triviaqa.jsonl -name-a Fused -name-b Cascade -name FusedVsCascade$(READER_NAME)TriviaQA -json-out eval/cmp-fused-cascade-$(READER_TAG)-triviaqa.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-musique.jsonl -b runs/cascade-$(READER_TAG)-musique.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveRAGVsCascade$(READER_NAME)MuSiQue -json-out eval/cmp-naive-cascade-$(READER_TAG)-musique.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-hotpotqa.jsonl -b runs/cascade-$(READER_TAG)-hotpotqa.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveRAGVsCascade$(READER_NAME)HotpotQA -json-out eval/cmp-naive-cascade-$(READER_TAG)-hotpotqa.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-2wiki.jsonl -b runs/cascade-$(READER_TAG)-2wiki.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveRAGVsCascade$(READER_NAME)TwoWiki -json-out eval/cmp-naive-cascade-$(READER_TAG)-2wiki.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-nq.jsonl -b runs/cascade-$(READER_TAG)-nq.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveRAGVsCascade$(READER_NAME)NQ -json-out eval/cmp-naive-cascade-$(READER_TAG)-nq.json
	go run ./cmd/compare -a runs/naive-$(READER_TAG)-triviaqa.jsonl -b runs/cascade-$(READER_TAG)-triviaqa.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveRAGVsCascade$(READER_NAME)TriviaQA -json-out eval/cmp-naive-cascade-$(READER_TAG)-triviaqa.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-musique.jsonl -b runs/cascade-$(READER_TAG)-musique.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascade$(READER_NAME)MuSiQue -json-out eval/cmp-rerank-cascade-$(READER_TAG)-musique.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-hotpotqa.jsonl -b runs/cascade-$(READER_TAG)-hotpotqa.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascade$(READER_NAME)HotpotQA -json-out eval/cmp-rerank-cascade-$(READER_TAG)-hotpotqa.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-2wiki.jsonl -b runs/cascade-$(READER_TAG)-2wiki.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascade$(READER_NAME)TwoWiki -json-out eval/cmp-rerank-cascade-$(READER_TAG)-2wiki.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-nq.jsonl -b runs/cascade-$(READER_TAG)-nq.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascade$(READER_NAME)NQ -json-out eval/cmp-rerank-cascade-$(READER_TAG)-nq.json
	go run ./cmd/compare -a runs/rerank-$(READER_TAG)-triviaqa.jsonl -b runs/cascade-$(READER_TAG)-triviaqa.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascade$(READER_NAME)TriviaQA -json-out eval/cmp-rerank-cascade-$(READER_TAG)-triviaqa.json
	go run ./cmd/result -suffix $(READER_TAG) -baselines closedbook,naive,rerank,fused -name Result$(READER_NAME) -json-out eval/result-$(READER_TAG).json
	go run ./cmd/cost -suffix $(READER_TAG) -architectures closedbook,naive,rerank,fused,cascade -name Cost$(READER_NAME) -json-out eval/cost-$(READER_TAG).json

# Needs the naive and closedbook runs.
diagnose:
	go run ./cmd/diagnose -dataset dataset/musique_dev.jsonl -run runs/naive-musique.jsonl -closedbook runs/closedbook-musique.jsonl -collection ragbench-wiki18 -top-k 5 -name DiagMuSiQue -json-out eval/diagnosis-musique.json
	go run ./cmd/diagnose -dataset dataset/hotpotqa_dev.jsonl -run runs/naive-hotpotqa.jsonl -closedbook runs/closedbook-hotpotqa.jsonl -collection ragbench-wiki18 -top-k 5 -name DiagHotpotQA -json-out eval/diagnosis-hotpotqa.json
	go run ./cmd/diagnose -dataset dataset/2wikimultihopqa_dev.jsonl -run runs/naive-2wiki.jsonl -closedbook runs/closedbook-2wiki.jsonl -collection ragbench-wiki18 -top-k 5 -name DiagTwoWiki -json-out eval/diagnosis-2wiki.json

# bm25, hybrid, hyde and rerank keep the 5-passage budget, so naive is the
# control. ircot, crag10 and neighbour do not, so they go against naive10/12.
compare:
	go run ./cmd/compare -a runs/closedbook-musique.jsonl -b runs/naive-musique.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveMuSiQue -json-out eval/cmp-closedbook-naive-musique.json
	go run ./cmd/compare -a runs/closedbook-hotpotqa.jsonl -b runs/naive-hotpotqa.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveHotpotQA -json-out eval/cmp-closedbook-naive-hotpotqa.json
	go run ./cmd/compare -a runs/closedbook-2wiki.jsonl -b runs/naive-2wiki.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveTwoWiki -json-out eval/cmp-closedbook-naive-2wiki.json
	go run ./cmd/compare -a runs/closedbook-nq.jsonl -b runs/naive-nq.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveNQ -json-out eval/cmp-closedbook-naive-nq.json
	go run ./cmd/compare -a runs/closedbook-triviaqa.jsonl -b runs/naive-triviaqa.jsonl -name-a ClosedBook -name-b NaiveRAG -name ClosedBookVsNaiveTriviaQA -json-out eval/cmp-closedbook-naive-triviaqa.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/bm25-musique.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveMuSiQue -json-out eval/cmp-naive-bm25-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/bm25-hotpotqa.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveHotpotQA -json-out eval/cmp-naive-bm25-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/bm25-2wiki.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveTwoWiki -json-out eval/cmp-naive-bm25-2wiki.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/hybrid-musique.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridMuSiQue -json-out eval/cmp-naive-hybrid-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/hybrid-hotpotqa.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridHotpotQA -json-out eval/cmp-naive-hybrid-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/hybrid-2wiki.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridTwoWiki -json-out eval/cmp-naive-hybrid-2wiki.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/hyde-musique.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDEMuSiQue -json-out eval/cmp-naive-hyde-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/hyde-hotpotqa.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDEHotpotQA -json-out eval/cmp-naive-hyde-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/hyde-2wiki.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDETwoWiki -json-out eval/cmp-naive-hyde-2wiki.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/rerank-musique.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankMuSiQue -json-out eval/cmp-naive-rerank-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/rerank-hotpotqa.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankHotpotQA -json-out eval/cmp-naive-rerank-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/rerank-2wiki.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankTwoWiki -json-out eval/cmp-naive-rerank-2wiki.json
	go run ./cmd/compare -a runs/naive10-musique.jsonl -b runs/ircot-musique.jsonl -name-a NaiveTen -name-b IRCoT -name NaiveTenVsIRCoTMuSiQue -json-out eval/cmp-naive10-ircot-musique.json
	go run ./cmd/compare -a runs/naive10-hotpotqa.jsonl -b runs/ircot-hotpotqa.jsonl -name-a NaiveTen -name-b IRCoT -name NaiveTenVsIRCoTHotpotQA -json-out eval/cmp-naive10-ircot-hotpotqa.json
	go run ./cmd/compare -a runs/naive10-2wiki.jsonl -b runs/ircot-2wiki.jsonl -name-a NaiveTen -name-b IRCoT -name NaiveTenVsIRCoTTwoWiki -json-out eval/cmp-naive10-ircot-2wiki.json
	go run ./cmd/compare -a runs/naive10-musique.jsonl -b runs/crag10-musique.jsonl -name-a NaiveTen -name-b CRAGTen -name NaiveTenVsCRAGTenMuSiQue -json-out eval/cmp-naive10-crag10-musique.json
	go run ./cmd/compare -a runs/naive10-hotpotqa.jsonl -b runs/crag10-hotpotqa.jsonl -name-a NaiveTen -name-b CRAGTen -name NaiveTenVsCRAGTenHotpotQA -json-out eval/cmp-naive10-crag10-hotpotqa.json
	go run ./cmd/compare -a runs/naive10-2wiki.jsonl -b runs/crag10-2wiki.jsonl -name-a NaiveTen -name-b CRAGTen -name NaiveTenVsCRAGTenTwoWiki -json-out eval/cmp-naive10-crag10-2wiki.json
	go run ./cmd/compare -a runs/naive12-musique.jsonl -b runs/neighbour-musique.jsonl -name-a NaiveTwelve -name-b Neighbour -name NaiveTwelveVsNeighbourMuSiQue -json-out eval/cmp-naive12-neighbour-musique.json
	go run ./cmd/compare -a runs/naive12-hotpotqa.jsonl -b runs/neighbour-hotpotqa.jsonl -name-a NaiveTwelve -name-b Neighbour -name NaiveTwelveVsNeighbourHotpotQA -json-out eval/cmp-naive12-neighbour-hotpotqa.json
	go run ./cmd/compare -a runs/naive12-2wiki.jsonl -b runs/neighbour-2wiki.jsonl -name-a NaiveTwelve -name-b Neighbour -name NaiveTwelveVsNeighbourTwoWiki -json-out eval/cmp-naive12-neighbour-2wiki.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/adaptive-musique.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveMuSiQue -json-out eval/cmp-naive-adaptive-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/adaptive-hotpotqa.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveHotpotQA -json-out eval/cmp-naive-adaptive-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/adaptive-2wiki.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveTwoWiki -json-out eval/cmp-naive-adaptive-2wiki.json
	go run ./cmd/compare -a runs/ircot-musique.jsonl -b runs/adaptive-musique.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveMuSiQue -json-out eval/cmp-ircot-adaptive-musique.json
	go run ./cmd/compare -a runs/ircot-hotpotqa.jsonl -b runs/adaptive-hotpotqa.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveHotpotQA -json-out eval/cmp-ircot-adaptive-hotpotqa.json
	go run ./cmd/compare -a runs/ircot-2wiki.jsonl -b runs/adaptive-2wiki.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveTwoWiki -json-out eval/cmp-ircot-adaptive-2wiki.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/bm25-nq.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveNQ -json-out eval/cmp-naive-bm25-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/bm25-triviaqa.jsonl -name-a NaiveRAG -name-b BMTwentyFive -name NaiveVsBMTwentyFiveTriviaQA -json-out eval/cmp-naive-bm25-triviaqa.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/hybrid-nq.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridNQ -json-out eval/cmp-naive-hybrid-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/hybrid-triviaqa.jsonl -name-a NaiveRAG -name-b Hybrid -name NaiveVsHybridTriviaQA -json-out eval/cmp-naive-hybrid-triviaqa.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/hyde-nq.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDENQ -json-out eval/cmp-naive-hyde-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/hyde-triviaqa.jsonl -name-a NaiveRAG -name-b HyDE -name NaiveVsHyDETriviaQA -json-out eval/cmp-naive-hyde-triviaqa.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/rerank-nq.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankNQ -json-out eval/cmp-naive-rerank-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/rerank-triviaqa.jsonl -name-a NaiveRAG -name-b Rerank -name NaiveVsRerankTriviaQA -json-out eval/cmp-naive-rerank-triviaqa.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/adaptive-nq.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveNQ -json-out eval/cmp-naive-adaptive-nq.json
	go run ./cmd/compare -a runs/ircot-nq.jsonl -b runs/adaptive-nq.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveNQ -json-out eval/cmp-ircot-adaptive-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/adaptive-triviaqa.jsonl -name-a NaiveRAG -name-b Adaptive -name NaiveVsAdaptiveTriviaQA -json-out eval/cmp-naive-adaptive-triviaqa.json
	go run ./cmd/compare -a runs/ircot-triviaqa.jsonl -b runs/adaptive-triviaqa.jsonl -name-a IRCoT -name-b Adaptive -name IRCoTVsAdaptiveTriviaQA -json-out eval/cmp-ircot-adaptive-triviaqa.json
	go run ./cmd/compare -a runs/rerank-musique.jsonl -b runs/cascade-musique.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeMuSiQue -json-out eval/cmp-rerank-cascade-musique.json
	go run ./cmd/compare -a runs/rerank-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeHotpotQA -json-out eval/cmp-rerank-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/rerank-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeTwoWiki -json-out eval/cmp-rerank-cascade-2wiki.json
	go run ./cmd/compare -a runs/rerank-nq.jsonl -b runs/cascade-nq.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeNQ -json-out eval/cmp-rerank-cascade-nq.json
	go run ./cmd/compare -a runs/rerank-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a Rerank -name-b Cascade -name RerankVsCascadeTriviaQA -json-out eval/cmp-rerank-cascade-triviaqa.json
	go run ./cmd/compare -a runs/naive-musique.jsonl -b runs/cascade-musique.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeMuSiQue -json-out eval/cmp-naive-cascade-musique.json
	go run ./cmd/compare -a runs/naive-hotpotqa.jsonl -b runs/cascade-hotpotqa.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeHotpotQA -json-out eval/cmp-naive-cascade-hotpotqa.json
	go run ./cmd/compare -a runs/naive-2wiki.jsonl -b runs/cascade-2wiki.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeTwoWiki -json-out eval/cmp-naive-cascade-2wiki.json
	go run ./cmd/compare -a runs/naive-nq.jsonl -b runs/cascade-nq.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeNQ -json-out eval/cmp-naive-cascade-nq.json
	go run ./cmd/compare -a runs/naive-triviaqa.jsonl -b runs/cascade-triviaqa.jsonl -name-a NaiveRAG -name-b Cascade -name NaiveVsCascadeTriviaQA -json-out eval/cmp-naive-cascade-triviaqa.json
