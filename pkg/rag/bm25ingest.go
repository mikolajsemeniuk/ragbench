package rag

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// BM25Ingester fills a lexical collection. It mirrors NaiveRAG.Ingest so that
// cmd/ingest can drive either one, but it calls no embedding server at all -
// building an inverted index is pure CPU work, which is what makes adding
// lexical search to an already indexed 21M-passage corpus cheap.
type BM25Ingester struct {
	Store      storage.SparseStore
	Collection string
	Encoder    *BM25

	// Empty counts passages that produced no terms at all and therefore could
	// not be indexed. Reported by the caller so the final point count can be
	// reconciled with the corpus instead of looking like data loss.
	Empty atomic.Int64
}

func NewBM25Ingester(store storage.SparseStore, collection string) *BM25Ingester {
	return &BM25Ingester{Store: store, Collection: collection, Encoder: NewBM25()}
}

func (r *BM25Ingester) Ingest(ctx context.Context, docs []Document) error {
	if len(docs) == 0 {
		return nil
	}

	points := make([]storage.Point, 0, len(docs))
	for _, d := range docs {
		sparse := r.Encoder.Document(d.Text)
		if sparse.Empty() {
			r.Empty.Add(1)
			continue
		}
		points = append(points, storage.Point{ID: d.ID, Text: d.Text, Sparse: sparse})
	}

	if err := r.Store.UpsertSparse(ctx, r.Collection, points); err != nil {
		return fmt.Errorf("upsert sparse points: %w", err)
	}
	return nil
}
