package provider

import (
	"context"
	"sync/atomic"
)

// Usage accumulates what one question cost in generation tokens.
//
// It travels on the context rather than through the Generator interface on
// purpose. An architecture like IRCoT or CRAG makes several generation calls
// per question through code that has no idea it is being measured; threading a
// counter through every signature would touch every architecture and every
// call site to collect a number none of them cares about. On the context it is
// opt-in, invisible to the pipelines, and correct for architectures that do not
// exist yet.
//
// Tokens per question are the only cost measure comparable across hardware. A
// latency of 41.7s against 8.3s says as much about the GPU and the concurrency
// as about the architecture; "IRCoT spends 3.4 generation calls and 5,800
// prompt tokens per question" is reproducible by anyone.
type Usage struct {
	PromptTokens     atomic.Int64
	CompletionTokens atomic.Int64
	Calls            atomic.Int64
}

type usageKey struct{}

// WithUsage returns a context that accumulates generation usage, and the
// accumulator to read once the work on that context is done.
func WithUsage(ctx context.Context) (context.Context, *Usage) {
	u := &Usage{}
	return context.WithValue(ctx, usageKey{}, u), u
}

// UsageFrom returns the accumulator carried by ctx, or nil when nothing is
// being measured.
func UsageFrom(ctx context.Context) *Usage {
	u, _ := ctx.Value(usageKey{}).(*Usage)
	return u
}

func recordUsage(ctx context.Context, prompt, completion int64) {
	u := UsageFrom(ctx)
	if u == nil {
		return
	}
	u.PromptTokens.Add(prompt)
	u.CompletionTokens.Add(completion)
	u.Calls.Add(1)
}
