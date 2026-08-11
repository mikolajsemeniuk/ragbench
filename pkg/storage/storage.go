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
	Search(ctx context.Context, collection string, vector []float32, limit int) ([]string, error)
}
