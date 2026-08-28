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

// RouteReporter is implemented by architectures that pick one of several
// strategies per question. The counts are what makes such a row in a results
// table readable: a router that sends every question down the same branch
// scores like that branch, and without the distribution there is no way to
// tell that apart from a router that is genuinely choosing.
type RouteReporter interface {
	Routes() map[string]int64
}
