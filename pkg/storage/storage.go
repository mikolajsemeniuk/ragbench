// Package storage holds the vector-store backends the benchmark can run
// against.
package storage

import "context"

type Point struct {
	ID     uint64
	Vector []float32
	Text   string
}

type VectorStore interface {
	EnsureCollection(ctx context.Context, collection string, vectorSize int) error
	Upsert(ctx context.Context, collection string, points []Point) error
	// Search returns the matching points (ID + text) ordered by decreasing
	// relevance. The IDs are needed both to build the prompt and to compute
	// the retrieval metrics (Recall@K, MRR).
	Search(ctx context.Context, collection string, vector []float32, limit int) ([]Point, error)
}
