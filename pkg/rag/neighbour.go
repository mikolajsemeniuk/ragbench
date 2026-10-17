package rag

import (
	"context"
	"fmt"

	"github.com/mikolajsemeniuk/ragbench/pkg/flashrag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// NeighbourRAG retrieves as NaiveRAG does and then adds, for every hit, the
// passages next to it in the same article.
//
// The motivation is that a Wikipedia article is chunked into ~100-word
// passages with no regard for where a fact ends. The retriever can land on the
// paragraph that discusses the right entity while the sentence carrying the
// answer sits in the chunk immediately after it.
//
// It is included to TEST that hypothesis rather than because the measurements
// support it. cmd/diagnose puts the "answer-bearing article was in the top-k
// but the wrong slice of it was taken" bucket at only 5.0% of failures on
// 2WikiMultihopQA, 6.7% on HotpotQA and 5.3% on MuSiQue - an order of
// magnitude below what the project's own notes assumed. Either this row shows
// a gain of that size and the diagnostic is validated, or it does not and the
// assumption was wrong; both are results worth reporting.
//
// Note on comparability: expanding k hits by radius r puts up to
// k * (2r + 1) passages in context, so this must be read against a NaiveRAG
// run with a matching -top-k, not against the 5-passage baseline. Otherwise
// the row measures context size, which naive10 already showed is worth
// +0.0221 Exact Match on 2WikiMultihopQA on its own.
type NeighbourRAG struct {
	Store      storage.VectorStore
	Fetcher    storage.PointFetcher
	Collection string

	Embedder  Embedder
	Generator Generator
	Index     *flashrag.TitleIndex

	// TopK is how many passages the retriever returns, Radius how far to
	// either side of each one to expand, MaxPassages the cap on the result.
	TopK        int
	Radius      int
	MaxPassages int

	QueryPrefix    string
	DocumentPrefix string
}

func NewNeighbourRAG(store storage.VectorStore, fetcher storage.PointFetcher, collection string, embedder Embedder, generator Generator, index *flashrag.TitleIndex) *NeighbourRAG {
	return &NeighbourRAG{
		Store:       store,
		Fetcher:     fetcher,
		Collection:  collection,
		Embedder:    embedder,
		Generator:   generator,
		Index:       index,
		TopK:        5,
		Radius:      1,
		MaxPassages: 15,
	}
}

func (r *NeighbourRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	vectors, err := r.Embedder.Embed(ctx, []string{r.QueryPrefix + question})
	if err != nil {
		return "", nil, fmt.Errorf("embed question: %w", err)
	}

	hits, err := r.Store.Search(ctx, r.Collection, vectors[0], r.TopK)
	if err != nil {
		return "", nil, fmt.Errorf("search: %w", err)
	}

	// Each hit is emitted before its own neighbours, and the hits keep their
	// retrieval order. That keeps the rank of every originally retrieved
	// passage unchanged, so a difference in MRR against the baseline reflects
	// what was added rather than a reshuffle of what was already there.
	seen := make(map[uint64]struct{}, r.MaxPassages)
	order := make([]uint64, 0, r.MaxPassages)
	texts := make(map[uint64]string, r.MaxPassages)
	var missing []uint64

	add := func(id uint64, text string) bool {
		if _, ok := seen[id]; ok {
			return true
		}
		if len(order) >= r.MaxPassages {
			return false
		}
		seen[id] = struct{}{}
		order = append(order, id)
		if text != "" {
			texts[id] = text
		} else {
			missing = append(missing, id)
		}
		return true
	}

outer:
	for _, hit := range hits {
		if !add(hit.ID, hit.Text) {
			break
		}
		title := flashrag.PassageTitle(hit.Text)
		for _, id := range r.Index.Neighbours(title, hit.ID, r.Radius) {
			if !add(id, "") {
				break outer
			}
		}
	}

	// The neighbours are known by id only; their text lives in the same
	// collection and is fetched in one round trip rather than one per
	// passage.
	if len(missing) > 0 {
		points, err := r.Fetcher.Retrieve(ctx, r.Collection, missing)
		if err != nil {
			return "", nil, fmt.Errorf("fetch neighbouring passages: %w", err)
		}
		for _, p := range points {
			texts[p.ID] = p.Text
		}
	}

	points := make([]storage.Point, 0, len(order))
	for _, id := range order {
		// A neighbour the index knows about but the collection does not is
		// dropped rather than fabricated as an empty passage.
		if text, ok := texts[id]; ok {
			points = append(points, storage.Point{ID: id, Text: text})
		}
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, points))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, points, nil
}
