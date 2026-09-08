package rag

import (
	"context"
	"fmt"
	"math"

	"github.com/mikolajsemeniuk/ragbench/pkg/metrics"
	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// CascadeRAG answers with a cheap strategy and escalates only the questions
// that strategy declined to answer. It is a loop over other architectures, not
// a retriever: the stages are whatever cmd/bench's -cascade-stages names, and
// the default first stage is FusedRAG (fused.go).
//
// The idea in one sentence: try cheaply, and only when the reader itself says
// the passages do not support an answer, try differently.
//
// Every adaptive system compared here decides how much work a question needs
// from a PRIOR signal - it asks the generator to predict difficulty from the
// question alone. AdaptiveRAG does exactly that and is worth +0.0004 Exact
// Match on 2WikiMultihopQA at 2.1 to 3.0 generation calls, sending 92% of
// that set down the single-hop branch and choosing the closed-book branch for
// no question at all (3 out of 3610 on NaturalQuestions). This cascade uses a
// POSTERIOR signal instead: the reader has already seen the retrieved passages
// and reported that they do not support an answer.
//
// The trigger is lossless, which is what makes escalation safe rather than a
// gamble. An abstention is a sentence and the gold answers in these datasets
// are short spans, so an abstention cannot score Exact Match 1 - verified over
// all 462,654 (question, architecture) pairs measured here, with zero
// exceptions. Escalating a declined question therefore cannot discard a correct
// answer; it can only move a question from certainly-wrong to possibly-right.
//
// Measured against the standard baseline with the same reader, on the paired
// Exact Match test: +0.0064 on NaturalQuestions (not significant), +0.0333 on
// TriviaQA, +0.0334 on HotpotQA, +0.0497 on 2WikiMultihopQA, +0.0244 on
// MuSiQue, at 1.16 to 1.58 generation calls per question. Against plain
// reranking it gains +0.1066 on NaturalQuestions and loses -0.0174 on
// HotpotQA: the escalation is worth +1 to +3 points on top of ANY first stage,
// while which first stage is best depends on the question set. See fused.go
// for why the default is the fused one.
//
// The abstention is blind to the other failure, a wrong answer given without
// hesitation, and that is where the remaining headroom is: an oracle choosing
// per question between naive retrieval, reranking and closed-book scores 0.403
// on 2WikiMultihopQA against the cascade's 0.278, and the abstention trigger
// collects 1.6 points of that gap. MinLogprob adds a second, continuous
// posterior signal for it: the mean log-probability of the answer's tokens,
// which the generator reports for free (provider.Confidence). A stage's
// answer is then also escalated when its mean log-probability falls below the
// threshold. Unlike the abstention this is NOT lossless - a correct answer
// given with low confidence is discarded, and only sometimes recovered by a
// later stage - so the threshold has to be chosen on held-out questions
// (cmd/sweep over the train splits) and the discard rate reported.
//
// Three caveats belong in any write-up. The property is exact for Exact Match
// and only approximate for token-level F1, where an abstention averages 0.0495
// and exceeds 0.5 for 0.09% of questions. The gain sits in the questions the
// first stage declined, not in the questions both systems answered - "EM (both
// answered)" against rerank is at or below zero on four of the five sets. And
// the abstention is detected heuristically - refusal phrasing, or an answer
// too long to be one of these datasets' spans - so the detector's threshold
// is a hyperparameter and belongs in an ablation.
type CascadeRAG struct {
	// Stages are tried in order. Names are parallel to Stages and are only
	// used to report which one answered.
	Stages []Pipeline
	Names  []string

	// AbstentionMaxWords is passed to metrics.IsAbstention. 0 leaves only the
	// explicit refusal phrases and disables the length signal.
	AbstentionMaxWords int

	// MinLogprob escalates an answer whose mean token log-probability is
	// below it, on top of the abstention trigger. Mean log-probabilities are
	// at most 0, so -Inf (the default) disables the signal and leaves the
	// lossless trigger alone. A provider that reports no log-probabilities
	// is treated the same way.
	MinLogprob float64
}

func NewCascadeRAG(names []string, stages []Pipeline, abstentionMaxWords int) *CascadeRAG {
	return &CascadeRAG{Stages: stages, Names: names, AbstentionMaxWords: abstentionMaxWords, MinLogprob: math.Inf(-1)}
}

// escalate reports whether a stage's answer should be handed to the next
// stage, and why: "abstained", "low-confidence", or "" to keep it.
func (r *CascadeRAG) escalate(ctx context.Context, answer string) string {
	if metrics.IsAbstention(answer, r.AbstentionMaxWords) {
		return "abstained"
	}
	if mean, ok := provider.ConfidenceFrom(ctx).Last(); ok && mean < r.MinLogprob {
		return "low-confidence"
	}
	return ""
}

func (r *CascadeRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	if len(r.Stages) == 0 {
		return "", nil, fmt.Errorf("cascade has no stages")
	}

	for i, stage := range r.Stages {
		answer, retrieved, err = stage.Query(ctx, question)
		if err != nil {
			return "", nil, fmt.Errorf("cascade stage %s: %w", r.Names[i], err)
		}
		// The last stage is kept whatever it says: there is nothing left to
		// escalate to, and a refusal is still the most honest answer
		// available.
		if i == len(r.Stages)-1 {
			TraceFrom(ctx).SetStage(r.Names[i], i+1)
			return answer, retrieved, nil
		}
		reason := r.escalate(ctx, answer)
		if reason == "" {
			TraceFrom(ctx).SetStage(r.Names[i], i+1)
			return answer, retrieved, nil
		}
		TraceFrom(ctx).AddEscalation(reason)
	}
	return answer, retrieved, nil
}
