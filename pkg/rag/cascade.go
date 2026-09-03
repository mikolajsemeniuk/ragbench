package rag

import (
	"context"
	"fmt"

	"github.com/mikolajsemeniuk/ragbench/pkg/metrics"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// CascadeRAG answers with a cheap strategy and escalates only the questions
// that strategy declined to answer. Its retrieval stage is FusedRAG, below.
//
// The two parts address the two things the measurements in this repository say
// about the baselines it is compared against.
//
// First, cross-encoder reranking is the strongest single retrieval improvement
// available (+0.0510 Exact Match on HotpotQA, +0.0388 on 2WikiMultihopQA,
// against the same five-passage budget) and it also fails badly enough to be
// unusable as a default: -0.1001 on NaturalQuestions. FusedRAG keeps the gain
// without the failure.
//
// Second, every adaptive system compared here decides how much work a question
// needs from a PRIOR signal - it asks the generator to predict difficulty from
// the question alone. AdaptiveRAG does exactly that and is worth +0.0004 Exact
// Match at 2.1 to 3.0 generation calls, sending 92% of 2WikiMultihopQA down the
// single-hop branch and choosing the closed-book branch for 3 questions out of
// 3610. This cascade uses a POSTERIOR signal instead: the reader has already
// seen the retrieved passages and reported that they do not support an answer.
// Worth +0.0164 at 1.29 calls.
//
// The trigger is lossless, which is what makes escalation safe rather than a
// gamble. An abstention is a sentence and the gold answers in these datasets
// are short spans, so an abstention cannot score Exact Match 1 - verified over
// all 298,096 (question, architecture) pairs measured here, with zero
// exceptions. Escalating a declined question therefore cannot discard a correct
// answer; it can only move a question from certainly-wrong to possibly-right.
//
// Two caveats belong in any write-up. The property is exact for Exact Match and
// only approximate for token-level F1, where an abstention averages 0.0487 and
// exceeds 0.5 for 0.09% of questions. And the abstention is detected
// heuristically - refusal phrasing, or an answer too long to be one of these
// datasets' spans - so the detector's threshold is a hyperparameter and belongs
// in an ablation.
type CascadeRAG struct {
	// Stages are tried in order. Names are parallel to Stages and are only
	// used to report which one answered.
	Stages []Pipeline
	Names  []string

	// AbstentionMaxWords is passed to metrics.IsAbstention. 0 leaves only the
	// explicit refusal phrases and disables the length signal.
	AbstentionMaxWords int
}

func NewCascadeRAG(names []string, stages []Pipeline, abstentionMaxWords int) *CascadeRAG {
	return &CascadeRAG{Stages: stages, Names: names, AbstentionMaxWords: abstentionMaxWords}
}

func (r *CascadeRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	if len(r.Stages) == 0 {
		return "", nil, fmt.Errorf("cascade has no stages")
	}

	for i, stage := range r.Stages {
		answer, retrieved, err = stage.Query(ctx, question)
		if err != nil {
			return "", nil, fmt.Errorf("cascade stage %s: %w", r.Names[i], err)
		}
		// The last stage is kept whatever it says: there is nothing left to
		// escalate to, and a refusal is still the most honest answer
		// available.
		if i == len(r.Stages)-1 || !metrics.IsAbstention(answer, r.AbstentionMaxWords) {
			TraceFrom(ctx).SetStage(r.Names[i], i+1)
			return answer, retrieved, nil
		}
	}
	return answer, retrieved, nil
}

// FusedRAG is the cascade's retrieval stage, and is registered as its own
// architecture so that the stage can be ablated on its own.
//
// It treats the cross-encoder as one voter among three rather than as the
// arbiter of the ranking. A cross-encoder scores question and passage jointly
// and therefore matches what the question is ASKING FOR, while a bi-encoder
// matches what it is ASKING ABOUT. For "where is the tv show the curse of oak
// island filmed" the cross-encoder promotes the three passages in the shortlist
// that state a filming location - for Stake Land, The Island and Come Outside -
// because each one answers "where was it filmed" perfectly. The passage naming
// the right show says nothing about filming locations and is pushed out of the
// top five. The failure is predictable from the question alone: with no
// capitalised word reranking costs -0.0475 Exact Match, with four or more it
// gains +0.0423.
//
// Fusing the rankings instead of replacing one with another keeps both signals.
// Reciprocal Rank Fusion is what makes the combination possible at all: a
// cosine similarity, a BM25 score and a cross-encoder logit are on three
// incomparable scales, but their ranks are not.
//
// Cost over plain reranking is one lexical search on the CPU. The generator is
// still called exactly once.
type FusedRAG struct {
	Store       storage.VectorStore
	SparseStore storage.SparseStore

	Collection       string
	SparseCollection string

	Embedder  Embedder
	Reranker  Reranker
	Generator Generator
	Encoder   *BM25

	// Candidates is how deep each retriever's list goes before fusion, TopK
	// how many survive it and reach the generator.
	Candidates int
	TopK       int
	RRFK       int

	QueryPrefix    string
	DocumentPrefix string
}

func NewFusedRAG(store storage.VectorStore, sparse storage.SparseStore, collection, sparseCollection string, embedder Embedder, reranker Reranker, generator Generator) *FusedRAG {
	return &FusedRAG{
		Store:            store,
		SparseStore:      sparse,
		Collection:       collection,
		SparseCollection: sparseCollection,
		Embedder:         embedder,
		Reranker:         reranker,
		Generator:        generator,
		Encoder:          NewBM25(),
		Candidates:       100,
		TopK:             5,
		RRFK:             60,
	}
}

func (r *FusedRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	points, err := r.retrieve(ctx, question)
	if err != nil {
		return "", nil, err
	}
	if len(points) == 0 {
		return "", nil, fmt.Errorf("no retriever returned a passage")
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, points))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, points, nil
}

func (r *FusedRAG) retrieve(ctx context.Context, question string) ([]storage.Point, error) {
	vectors, err := r.Embedder.Embed(ctx, []string{r.QueryPrefix + question})
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}

	dense, err := r.Store.Search(ctx, r.Collection, vectors[0], r.Candidates)
	if err != nil {
		return nil, fmt.Errorf("dense search: %w", err)
	}
	if len(dense) == 0 {
		return nil, nil
	}

	// The lexical query is built from the raw question, never the prefixed
	// one: an embedding-model instruction would only add noise terms to an
	// inverted index.
	lexical, err := r.SparseStore.SearchSparse(ctx, r.SparseCollection, r.Encoder.Query(question), r.Candidates)
	if err != nil {
		return nil, fmt.Errorf("lexical search: %w", err)
	}

	// The cross-encoder reorders the dense shortlist. The full ordering is
	// requested rather than a top-n, because fusion needs a rank for every
	// candidate, not just for the winners.
	documents := make([]string, len(dense))
	for i, d := range dense {
		documents[i] = d.Text
	}
	order, err := r.Reranker.Rerank(ctx, question, documents, len(documents))
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}
	reranked := make([]storage.Point, 0, len(order))
	for _, idx := range order {
		reranked = append(reranked, dense[idx])
	}

	// Three rankings over one pool. A passage the dense retriever found is
	// voted for twice - once for where the bi-encoder put it, once for where
	// the cross-encoder put it - while a passage only the lexical index found
	// is voted for once. That asymmetry is deliberate: the lexical list is
	// there to rescue passages dense retrieval missed, not to outvote it.
	return FuseRRF([][]storage.Point{dense, reranked, lexical}, r.RRFK, r.TopK), nil
}
