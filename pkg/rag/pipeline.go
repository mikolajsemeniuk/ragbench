package rag

import (
	"context"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// Pipeline is the contract every compared RAG architecture implements: given a
// question, return an answer and the passages that were in context when it was
// produced.
//
// It lives here as well as in cmd/bench because AdaptiveRAG is built out of
// other architectures and needs to name the type it delegates to.
type Pipeline interface {
	Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error)
}
