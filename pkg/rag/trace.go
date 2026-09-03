package rag

import (
	"context"
	"sync"
)

// Trace records what an architecture actually did on one question.
//
// Aggregate scores cannot explain a null result. CRAG ends with exactly five
// passages on 87.8% of 2WikiMultihopQA questions, but the "every passage was
// judged relevant" branch and the "no passage was, so the retrieval was
// replaced" branch both produce five - so the dumps cannot say which one
// fired, and the paper cannot say why the corrective step does not pay off.
//
// Like provider.Usage it travels on the context instead of the Pipeline
// signature, so that architectures stay unaware of being measured and the
// contract every one of them implements does not grow a field for each new
// method's internals. Every setter tolerates a nil receiver, so an
// architecture can record unconditionally whether or not anyone is listening.
type Trace struct {
	mu sync.Mutex
	TraceData
}

// TraceData is the trace without its lock, so that a snapshot can be copied,
// stored and marshalled freely.
type TraceData struct {
	// Route is the branch a routing architecture chose.
	Route string

	// Branch is CRAG's verdict on the initial retrieval, GradedKept how many
	// of the retrieved passages its grader judged useful, and RewrittenQuery
	// the query it searched with when it corrected.
	Branch         string
	GradedKept     int
	RewrittenQuery string

	// Steps is how many reasoning rounds IRCoT actually ran, out of MaxSteps,
	// and Reasoning the sentences it produced.
	Steps     int
	Reasoning []string

	// Stage is the name of the cascade stage whose answer was kept, and
	// StagesRun how many stages had to run to get it. Together they are the
	// cascade's cost profile: a stage that never fires costs nothing, and a
	// stage that always fires is not a cascade.
	Stage     string
	StagesRun int
}

type traceKey struct{}

// WithTrace returns a context that collects a trace, and the trace to read
// once the question has been answered.
func WithTrace(ctx context.Context) (context.Context, *Trace) {
	t := &Trace{}
	return context.WithValue(ctx, traceKey{}, t), t
}

// TraceFrom returns the trace carried by ctx, or nil when nothing is being
// recorded. The returned value is safe to call setters on either way.
func TraceFrom(ctx context.Context) *Trace {
	t, _ := ctx.Value(traceKey{}).(*Trace)
	return t
}

func (t *Trace) SetRoute(route string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.Route = route
	t.mu.Unlock()
}

func (t *Trace) SetGrade(kept int) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.GradedKept = kept
	t.mu.Unlock()
}

func (t *Trace) SetBranch(branch string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.Branch = branch
	t.mu.Unlock()
}

func (t *Trace) SetRewrittenQuery(query string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.RewrittenQuery = query
	t.mu.Unlock()
}

func (t *Trace) SetStage(name string, run int) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.Stage = name
	t.StagesRun = run
	t.mu.Unlock()
}

func (t *Trace) AddReasoning(sentence string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.Reasoning = append(t.Reasoning, sentence)
	t.Steps++
	t.mu.Unlock()
}

// Snapshot returns a copy safe to read after the question is done.
func (t *Trace) Snapshot() TraceData {
	if t == nil {
		return TraceData{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	data := t.TraceData
	data.Reasoning = append([]string(nil), t.Reasoning...)
	return data
}
