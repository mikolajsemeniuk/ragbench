package rag

import (
	"context"
	"fmt"
	"slices"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// HybridRAG retrieves with a dense retriever and a lexical one and fuses the
// two rankings with Reciprocal Rank Fusion.
//
// The two retrievers fail in opposite ways. The dense one understands
// paraphrase but blurs rare surface forms, so it misses a passage that the
// question names outright. The lexical one matches names exactly but is blind
// to a question that never uses the passage's vocabulary. Fusing them recovers
// a passage that either one found, which is why hybrid retrieval is the
// baseline every retrieval paper is expected to report.
//
// With Dense disabled the same type is the lexical-only (BM25) baseline. That
// is deliberate: running BM25 alone through exactly the same fusion, prompt
// and scoring code as the hybrid keeps the difference between the two rows in
// the results table equal to "was the dense list fused in", and nothing else.
//
// Fusion is Reciprocal Rank Fusion (Cormack et al., 2009):
//
//	score(d) = sum over lists of 1 / (k + rank(d))
//
// It combines rankings, not scores, which is what makes it usable here at all:
// a cosine similarity and a BM25 score live on different scales and are not
// comparable, while their ranks are. k = 60 is the value the paper proposes
// and the one every implementation uses; it damps the influence of the top
// position enough that one retriever cannot dominate the fusion on its own.
type HybridRAG struct {
	Store       storage.VectorStore
	SparseStore storage.SparseStore

	// Collection holds the dense vectors, SparseCollection the lexical index.
	// They are separate collections over the same corpus and the same passage
	// ids, which is what lets the fusion match documents across the two lists.
	Collection       string
	SparseCollection string

	Embedder  Embedder
	Generator Generator
	Encoder   *BM25

	// Candidates is how deep each retriever's list goes before fusion, TopK
	// how many survive it and reach the generator. TopK matches the baseline's
	// context budget, so a difference in scores measures which passages were
	// chosen rather than how many the generator was given.
	Candidates int
	TopK       int
	RRFK       int

	// Dense selects whether the dense list takes part. False makes this the
	// lexical-only baseline.
	Dense bool

	QueryPrefix    string
	DocumentPrefix string
}

func NewHybridRAG(store storage.VectorStore, sparse storage.SparseStore, collection, sparseCollection string, embedder Embedder, generator Generator) *HybridRAG {
	return &HybridRAG{
		Store:            store,
		SparseStore:      sparse,
		Collection:       collection,
		SparseCollection: sparseCollection,
		Embedder:         embedder,
		Generator:        generator,
		Encoder:          NewBM25(),
		Candidates:       50,
		TopK:             5,
		RRFK:             60,
		Dense:            true,
	}
}

// NewBM25RAG is the lexical-only baseline: the same pipeline with the dense
// list switched off.
func NewBM25RAG(sparse storage.SparseStore, sparseCollection string, generator Generator) *HybridRAG {
	return &HybridRAG{
		SparseStore:      sparse,
		SparseCollection: sparseCollection,
		Generator:        generator,
		Encoder:          NewBM25(),
		Candidates:       50,
		TopK:             5,
		RRFK:             60,
		Dense:            false,
	}
}

func (r *HybridRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	var lists [][]storage.Point

	if r.Dense {
		vectors, err := r.Embedder.Embed(ctx, []string{r.QueryPrefix + question})
		if err != nil {
			return "", nil, fmt.Errorf("embed question: %w", err)
		}
		dense, err := r.Store.Search(ctx, r.Collection, vectors[0], r.Candidates)
		if err != nil {
			return "", nil, fmt.Errorf("dense search: %w", err)
		}
		lists = append(lists, dense)
	}

	// The lexical query is built from the raw question, never from the
	// prefixed one: the instruction prefix is an input to the embedding model
	// and would only add noise terms to an inverted index.
	lexical, err := r.SparseStore.SearchSparse(ctx, r.SparseCollection, r.Encoder.Query(question), r.Candidates)
	if err != nil {
		return "", nil, fmt.Errorf("lexical search: %w", err)
	}
	lists = append(lists, lexical)

	points := FuseRRF(lists, r.RRFK, r.TopK)
	if len(points) == 0 {
		return "", nil, fmt.Errorf("neither retriever returned a passage")
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, points))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, points, nil
}

// FuseRRF merges ranked lists by Reciprocal Rank Fusion and returns the best
// limit documents.
//
// Ties are broken by best rank achieved in any list and then by id, so the
// output is fully deterministic: two runs over the same lists produce the same
// passages in the same order, which a benchmark result has to be able to
// promise.
func FuseRRF(lists [][]storage.Point, k, limit int) []storage.Point {
	if k <= 0 {
		k = 60
	}

	type entry struct {
		point    storage.Point
		score    float64
		bestRank int
	}
	byID := make(map[uint64]*entry)
	var order []uint64

	for _, list := range lists {
		for rank, p := range list {
			e, ok := byID[p.ID]
			if !ok {
				e = &entry{point: p, bestRank: rank}
				byID[p.ID] = e
				order = append(order, p.ID)
			}
			e.score += 1 / float64(k+rank+1)
			e.bestRank = min(e.bestRank, rank)
		}
	}

	slices.SortStableFunc(order, func(a, b uint64) int {
		x, y := byID[a], byID[b]
		switch {
		case x.score > y.score:
			return -1
		case x.score < y.score:
			return 1
		case x.bestRank != y.bestRank:
			return x.bestRank - y.bestRank
		case a < b:
			return -1
		case a > b:
			return 1
		}
		return 0
	})

	if limit > len(order) {
		limit = len(order)
	}
	out := make([]storage.Point, 0, limit)
	for _, id := range order[:limit] {
		out = append(out, byID[id].point)
	}
	return out
}
