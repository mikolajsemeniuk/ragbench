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
	// Search zwraca trafione punkty (ID + tekst) posortowane wg malejącej
	// trafności, potrzebne zarówno do budowy promptu, jak i do metryk
	// jakości retrievalu (Recall@K, MRR), które wymagają ID dokumentów.
	Search(ctx context.Context, collection string, vector []float32, limit int) ([]Point, error)
}
