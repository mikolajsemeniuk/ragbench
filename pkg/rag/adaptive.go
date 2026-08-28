package rag

import (
	"context"
	"fmt"
	"maps"
	"strings"
	"sync"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// Route names, used both to select a branch and as the keys of the reported
// distribution.
const (
	RouteClosedBook = "closedbook"
	RouteSingleHop  = "single"
	RouteMultiHop   = "multi"
)

// AdaptiveRAG picks a retrieval strategy per question (Jeong et al.,
// https://arxiv.org/abs/2403.14403).
//
// Every other architecture here treats all questions alike, and the
// measurements say that is wrong in both directions. On TriviaQA the generator
// already knows 42.5% of the answers without any retrieval, so running IRCoT
// over them spends up to six generation calls to arrive where one would have
// done. On MuSiQue a single query reaches the answer-bearing article for barely
// a fifth of the questions, so answering from one retrieval round is hopeless.
//
// The size of the prize is measured in cmd/diagnose: an oracle that picked,
// per question, the better of closed-book and NaiveRAG would gain +0.1026
// Exact Match on 2WikiMultihopQA and +0.0546 on HotpotQA - an order of
// magnitude more than the +0.0112 IRCoT gains over the matched-budget
// baseline. This architecture is the training-free attempt to collect part of
// it, and the baseline the paper's own router has to beat.
//
// Deviation from the paper, stated because it matters: the paper trains a
// small classifier on silver labels derived from which strategy actually
// succeeded. Here the generator classifies the question itself, in one call.
// A trained classifier would confound the comparison with an extra model, the
// same reason the CRAG implementation grades with the generator rather than a
// trained evaluator.
type AdaptiveRAG struct {
	Generator Generator

	// The three branches. ClosedBook answers from parametric knowledge,
	// SingleHop is one retrieval round, MultiHop is the iterative one. They
	// are supplied fully configured, so the router adds exactly one
	// classification call on top of whatever the chosen branch costs.
	ClosedBook Pipeline
	SingleHop  Pipeline
	MultiHop   Pipeline

	mu     sync.Mutex
	routes map[string]int64
}

func NewAdaptiveRAG(generator Generator, closedBook, singleHop, multiHop Pipeline) *AdaptiveRAG {
	return &AdaptiveRAG{
		Generator:  generator,
		ClosedBook: closedBook,
		SingleHop:  singleHop,
		MultiHop:   multiHop,
		routes:     make(map[string]int64),
	}
}

func (r *AdaptiveRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	verdict, err := r.Generator.Generate(ctx, buildRoutePrompt(question))
	if err != nil {
		return "", nil, fmt.Errorf("classify question: %w", err)
	}

	route := parseRoute(verdict)
	r.mu.Lock()
	r.routes[route]++
	r.mu.Unlock()

	var branch Pipeline
	switch route {
	case RouteClosedBook:
		branch = r.ClosedBook
	case RouteMultiHop:
		branch = r.MultiHop
	default:
		branch = r.SingleHop
	}

	answer, retrieved, err = branch.Query(ctx, question)
	if err != nil {
		return "", nil, fmt.Errorf("route %s: %w", route, err)
	}
	return answer, retrieved, nil
}

// Routes returns how many questions went down each branch.
func (r *AdaptiveRAG) Routes() map[string]int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return maps.Clone(r.routes)
}

// parseRoute reads the classifier's reply as one of three labels. Anything
// unparseable becomes the single-hop route, which is the safe default: it is
// the middle option, so a misread costs at most one retrieval round in either
// direction, whereas defaulting to closed-book would silently turn the router
// into a no-retrieval system on every parse failure.
func parseRoute(reply string) string {
	for _, r := range reply {
		switch r {
		case 'A', 'a':
			return RouteClosedBook
		case 'B', 'b':
			return RouteSingleHop
		case 'C', 'c':
			return RouteMultiHop
		}
	}
	return RouteSingleHop
}

// buildRoutePrompt asks for one letter.
//
// The labels are described by what the question needs, not by what the system
// would do with it: a 7B model classifies "does answering this require
// combining two separate facts" far more reliably than "should I use IRCoT".
func buildRoutePrompt(question string) string {
	var b strings.Builder
	b.WriteString("Classify how much looking up the question needs.\n\n")
	fmt.Fprintf(&b, "Question: %s\n\n", question)
	b.WriteString("A - a widely known fact that needs no source, such as a famous person, country or date.\n")
	b.WriteString("B - one fact that has to be looked up in a single document.\n")
	b.WriteString("C - several facts from different documents that have to be combined, ")
	b.WriteString("for example when the question refers to an entity it does not name.\n\n")
	b.WriteString("Reply with only the letter A, B or C.")
	return b.String()
}
