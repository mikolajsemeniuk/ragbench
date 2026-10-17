package provider

import (
	"context"
	"math"
	"sync"
)

// Confidence carries the generator's own certainty about the last answer it
// produced on this context: the mean log-probability of the answer's tokens
// under greedy decoding.
//
// It exists for the cascade. An abstention is a binary posterior signal - the
// reader looked at the passages and said they do not support an answer - and
// it is lossless under Exact Match, but it is blind to the other failure: an
// answer that is wrong and given without hesitation. The oracle that picks
// per question between naive retrieval, reranking and closed-book scores
// 0.403 Exact Match on 2WikiMultihopQA against the cascade's 0.278, and the
// abstention trigger collects 1.6 points of that gap. The rest sits in
// confident wrong answers, and the only free signal about them is how
// probable the model itself found the tokens it emitted.
//
// Like Usage it travels on the context, so that architectures stay unaware of
// being measured and the Generator interface does not grow a field for it.
// Only the LAST generation is kept, because every architecture ends with the
// answer call, and that is the one whose certainty is meaningful; a HyDE draft
// or an IRCoT reasoning sentence generated before it is overwritten.
//
// A provider that cannot report log-probabilities never records, and Last
// then reports ok=false, which the cascade reads as "no confidence signal" and
// falls back to the abstention trigger alone.
type Confidence struct {
	mu     sync.Mutex
	mean   float64
	tokens int
	ok     bool
}

type confidenceKey struct{}

// WithConfidence returns a context that records the confidence of each
// generation made on it, and the holder to read once the work is done.
func WithConfidence(ctx context.Context) (context.Context, *Confidence) {
	c := &Confidence{}
	return context.WithValue(ctx, confidenceKey{}, c), c
}

// ConfidenceFrom returns the holder carried by ctx, or nil when nothing is
// being recorded. The returned value is safe to call methods on either way.
func ConfidenceFrom(ctx context.Context) *Confidence {
	c, _ := ctx.Value(confidenceKey{}).(*Confidence)
	return c
}

// Last returns the mean token log-probability of the most recent generation
// and whether one was recorded at all.
func (c *Confidence) Last() (mean float64, ok bool) {
	if c == nil {
		return math.NaN(), false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.mean, c.ok
}

func recordConfidence(ctx context.Context, logprobs []float64) {
	c := ConfidenceFrom(ctx)
	if c == nil || len(logprobs) == 0 {
		return
	}
	sum := 0.0
	for _, lp := range logprobs {
		sum += lp
	}
	c.mu.Lock()
	c.mean = sum / float64(len(logprobs))
	c.tokens = len(logprobs)
	c.ok = true
	c.mu.Unlock()
}
