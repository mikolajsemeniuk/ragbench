package rag

import (
	"context"
	"fmt"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// FusedRAG is the cascade's default first stage, and is registered as its own
// architecture so that the stage can be measured without the escalation.
//
// It treats the cross-encoder as one voter among three rather than as the
// arbiter of the ranking. A cross-encoder scores question and passage jointly
// and therefore matches what the question is ASKING FOR, while a bi-encoder
// matches what it is ASKING ABOUT. For "where is the tv show the curse of oak
// island filmed" the cross-encoder promotes the three passages in the shortlist
// that state a filming location - for Stake Land, The Island and Come Outside -
// because each one answers "where was it filmed" perfectly. The passage naming
// the right show says nothing about filming locations and is pushed out of the
// top five. That is how plain reranking loses -0.1001 Exact Match on
// NaturalQuestions while gaining +0.0510 on HotpotQA.
//
// Fusing the rankings instead of replacing one with another keeps both signals.
// Reciprocal Rank Fusion is what makes the combination possible at all: a
// cosine similarity, a BM25 score and a cross-encoder logit are on three
// incomparable scales, but their ranks are not.
//
// What the fusion buys, measured: it is a compromise, not a free lunch. As the
// cascade's first stage it scores 0.3253 Exact Match on NaturalQuestions
// (rerank 0.2352, naive 0.3353) and 0.3514 on HotpotQA (rerank 0.3757, naive
// 0.3249) - it avoids the collapse but gives back part of the gain. Its case
// is robustness: of the first stages measured, it has the best mean over the
// five question sets and the smallest worst-case loss against the best cascade
// on each set (-0.025, on HotpotQA; rerank first loses -0.052 on NQ).
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
