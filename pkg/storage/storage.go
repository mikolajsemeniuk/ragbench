// Package storage holds the vector-store backends the benchmark can run
// against.
package storage

import "context"

// SparseVector is a bag-of-terms vector: Indices holds the term ids that occur
// in the text, Values their weights. It is the representation a lexical
// (BM25) index needs, and it is stored alongside - not instead of - the dense
// vector produced by the embedding model.
//
// Indices are 32-bit because that is what Qdrant's sparse index accepts.
type SparseVector struct {
	Indices []uint32
	Values  []float32
}

func (v SparseVector) Empty() bool { return len(v.Indices) == 0 }

type Point struct {
	ID     uint64
	Vector []float32
	Text   string

	// Sparse is set only for points in a lexical collection. A point never
	// carries both: the dense and the sparse corpora live in separate
	// collections so that adding lexical search does not require re-embedding
	// the 21M passages that are already indexed.
	Sparse SparseVector
}

type VectorStore interface {
	EnsureCollection(ctx context.Context, collection string, vectorSize int) error
	Upsert(ctx context.Context, collection string, points []Point) error
	// Search returns the matching points (ID + text) ordered by decreasing
	// relevance. The IDs are needed both to build the prompt and to compute
	// the retrieval metrics (Recall@K, MRR).
	Search(ctx context.Context, collection string, vector []float32, limit int) ([]Point, error)
}

// SparseStore is the lexical half of the hybrid retriever. It is a separate
// interface rather than extra methods on VectorStore because the dense
// collection built by the original ingest has no sparse vectors and never
// will - the two are queried side by side and fused by the caller.
type SparseStore interface {
	EnsureSparseCollection(ctx context.Context, collection string) error
	UpsertSparse(ctx context.Context, collection string, points []Point) error
	SearchSparse(ctx context.Context, collection string, vector SparseVector, limit int) ([]Point, error)
}

// PointFetcher retrieves points by id. Neighbour expansion needs it: once a
// passage is retrieved, the passages next to it in the same article are known
// by id and only their text is missing.
type PointFetcher interface {
	Retrieve(ctx context.Context, collection string, ids []uint64) ([]Point, error)
}
