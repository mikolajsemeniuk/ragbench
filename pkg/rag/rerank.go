package rag

import (
	"context"
	"fmt"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// Reranker reorders a shortlist of documents by relevance to the query,
// returning the indices of the best n.
type Reranker interface {
	Rerank(ctx context.Context, query string, documents []string, n int) ([]int, error)
}

// RerankRAG retrieves a deep shortlist with the fast bi-encoder, reorders it
// with a slow cross-encoder, and hands the generator the top few.
//
// It is aimed at one measured failure mode. cmd/diagnose shows that for the
// questions NaiveRAG gets wrong, the article that actually states the answer
// is already inside the retriever's top 100 in 15.7% (2WikiMultihopQA), 32.7%
// (HotpotQA) and 17.3% (MuSiQue) of cases - found, but ranked below the cut.
// Nothing that reasons over the top 5 can recover those; the fix has to change
// which 5 are chosen.
//
// Unlike IRCoT and CRAG this adds no generator calls, so the extra cost is one
// cross-encoder pass rather than several LLM round-trips.
type RerankRAG struct {
	Store      storage.VectorStore
	Collection string

	Embedder  Embedder
	Reranker  Reranker
	Generator Generator

	// Candidates is how deep the bi-encoder shortlist goes, TopK how many
	// survive reranking and reach the generator. TopK matches the baseline's
	// context budget, so a difference in scores measures ranking quality
	// rather than how much text the generator was given.
	Candidates int
	TopK       int

	QueryPrefix    string
	DocumentPrefix string
}

func NewRerankRAG(store storage.VectorStore, collection string, embedder Embedder, reranker Reranker, generator Generator) *RerankRAG {
	return &RerankRAG{
		Store:      store,
		Collection: collection,
		Embedder:   embedder,
		Reranker:   reranker,
		Generator:  generator,
		Candidates: 100,
		TopK:       5,
	}
}

func (r *RerankRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	vectors, err := r.Embedder.Embed(ctx, []string{r.QueryPrefix + question})
	if err != nil {
		return "", nil, fmt.Errorf("embed question: %w", err)
	}

	candidates, err := r.Store.Search(ctx, r.Collection, vectors[0], r.Candidates)
	if err != nil {
		return "", nil, fmt.Errorf("search: %w", err)
	}
	if len(candidates) == 0 {
		return "", nil, fmt.Errorf("search returned no candidates")
	}

	documents := make([]string, len(candidates))
	for i, c := range candidates {
		documents[i] = c.Text
	}
	order, err := r.Reranker.Rerank(ctx, question, documents, r.TopK)
	if err != nil {
		return "", nil, fmt.Errorf("rerank: %w", err)
	}

	points := make([]storage.Point, 0, len(order))
	for _, idx := range order {
		points = append(points, candidates[idx])
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, points))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, points, nil
}
