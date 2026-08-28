package rag

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// CRAG implements Corrective RAG (Yan et al.,
// https://arxiv.org/abs/2401.15884) in a training-free form.
//
// NaiveRAG hands the generator whatever the retriever returned and hopes it is
// relevant. The baseline measurements say how often that hope is misplaced: a
// gold answer appears anywhere in the retrieved context for only 34% of
// 2WikiMultihopQA questions and 23% of MuSiQue's. CRAG adds the missing step -
// judge the retrieval before using it, and act when it is bad.
//
// This is not the full method and must not be labelled "CRAG" in a results
// table. Call it "CRAG (offline, corpus-only)". Three deviations, all
// deliberate:
//
//   - The paper trains a T5 retrieval evaluator. Here the generator itself
//     judges relevance, which keeps the comparison against NaiveRAG and IRCoT
//     controlled: same corpus, same retriever, same generator, only the
//     architecture differs. A trained evaluator would confound the comparison
//     with an extra model.
//   - The paper's corrective action for bad retrieval is web search. There is
//     no web here - the corpus is fixed, which is what makes the comparison
//     reproducible - so the correction is to rewrite the query and search the
//     same corpus again. This is the standard offline substitution, and it
//     bounds this variant by what the corpus contains.
//   - The paper's knowledge refinement step, which decomposes each retrieved
//     passage into strips and filters them, is absent entirely. Passages are
//     used whole.
//
// The three branches of the paper's corrective step, recorded per question in
// the Trace. Which one fires is the only thing that explains a CRAG result,
// and it cannot be recovered from the passage count: "every passage relevant"
// and "no passage relevant, retrieval replaced" both end with TopK passages.
const (
	BranchCorrect   = "correct"
	BranchIncorrect = "incorrect"
	BranchAmbiguous = "ambiguous"
)

type CRAG struct {
	Store      storage.VectorStore
	Collection string

	Embedder  Embedder
	Generator Generator

	// Stepper grades the retrieval and rewrites the query. Separate from
	// Generator for the same reason as in IRCoT: a query rewrite truncated at
	// the answer's token cap is a broken query, and it is then searched with.
	// Nil falls back to Generator.
	Stepper Generator

	TopK        int
	MaxPassages int

	QueryPrefix    string
	DocumentPrefix string
}

func NewCRAG(store storage.VectorStore, collection string, embedder Embedder, generator Generator) *CRAG {
	return &CRAG{
		Store:       store,
		Collection:  collection,
		Embedder:    embedder,
		Generator:   generator,
		TopK:        5,
		MaxPassages: 10,
	}
}

// Query retrieves, grades what came back, and corrects when the grade is poor.
//
// The three branches follow the paper. Correct: every passage was judged
// relevant, use them. Incorrect: none was, so the retrieval is discarded
// entirely and replaced by a rewritten query's results - keeping known-bad
// passages would only dilute the context. Ambiguous: some were, so the
// relevant ones are kept and topped up from a rewritten query, because a
// partial answer plus a second angle beats either alone.
func (r *CRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	initial, err := r.retrieve(ctx, question)
	if err != nil {
		return "", nil, fmt.Errorf("initial retrieve: %w", err)
	}

	relevant, err := r.grade(ctx, question, initial)
	if err != nil {
		return "", nil, fmt.Errorf("grade retrieval: %w", err)
	}

	trace := TraceFrom(ctx)
	trace.SetGrade(len(relevant))

	var final []storage.Point
	switch {
	case len(relevant) == len(initial):
		trace.SetBranch(BranchCorrect)
		final = relevant
	case len(relevant) == 0:
		trace.SetBranch(BranchIncorrect)
		corrected, err := r.correct(ctx, question)
		if err != nil {
			return "", nil, err
		}
		final = corrected
	default:
		trace.SetBranch(BranchAmbiguous)
		corrected, err := r.correct(ctx, question)
		if err != nil {
			return "", nil, err
		}
		final = merge(relevant, corrected, r.MaxPassages)
	}

	// A rewrite that returns nothing must not leave the generator with an
	// empty context - falling back to the original retrieval is strictly
	// better than answering blind.
	if len(final) == 0 {
		final = initial
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, final))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, final, nil
}

// correct rewrites the query and searches the same corpus again.
func (r *CRAG) correct(ctx context.Context, question string) ([]storage.Point, error) {
	rewritten, err := r.stepper().Generate(ctx, buildRewritePrompt(question))
	if err != nil {
		return nil, fmt.Errorf("rewrite query: %w", err)
	}
	rewritten = strings.TrimSpace(strings.Trim(strings.TrimSpace(rewritten), "\"'"))
	TraceFrom(ctx).SetRewrittenQuery(rewritten)
	if rewritten == "" {
		return nil, nil
	}

	points, err := r.retrieve(ctx, rewritten)
	if err != nil {
		return nil, fmt.Errorf("retrieve rewritten query: %w", err)
	}
	return points, nil
}

// grade asks the generator which passages actually help, in one call for the
// whole set rather than one per passage - the passages are graded relative to
// each other, and it keeps the cost at one call instead of TopK.
func (r *CRAG) grade(ctx context.Context, question string, points []storage.Point) ([]storage.Point, error) {
	if len(points) == 0 {
		return nil, nil
	}

	verdict, err := r.stepper().Generate(ctx, buildGradePrompt(question, points))
	if err != nil {
		return nil, err
	}

	keep := parseIndices(verdict, len(points))
	relevant := make([]storage.Point, 0, len(keep))
	for _, i := range keep {
		relevant = append(relevant, points[i])
	}
	return relevant, nil
}

// stepper returns the generator to use for an intermediate call.
func (r *CRAG) stepper() Generator {
	if r.Stepper != nil {
		return r.Stepper
	}
	return r.Generator
}

func (r *CRAG) retrieve(ctx context.Context, query string) ([]storage.Point, error) {
	vectors, err := r.Embedder.Embed(ctx, []string{r.QueryPrefix + query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	points, err := r.Store.Search(ctx, r.Collection, vectors[0], r.TopK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	return points, nil
}

// merge concatenates two passage sets, dropping duplicates and capping the
// result.
func merge(a, b []storage.Point, cap int) []storage.Point {
	seen := make(map[uint64]struct{}, len(a)+len(b))
	out := make([]storage.Point, 0, cap)
	for _, set := range [][]storage.Point{a, b} {
		for _, p := range set {
			if _, ok := seen[p.ID]; ok {
				continue
			}
			if len(out) >= cap {
				return out
			}
			seen[p.ID] = struct{}{}
			out = append(out, p)
		}
	}
	return out
}

// parseIndices reads the grader's reply as a list of 1-based document numbers.
// Anything it cannot parse yields no indices, which routes the question to the
// corrective branch - the safe direction, since an unreadable grade is not
// evidence that the retrieval was good.
func parseIndices(reply string, n int) []int {
	if strings.Contains(strings.ToLower(reply), "none") {
		return nil
	}

	seen := make(map[int]struct{})
	var out []int
	for _, field := range strings.FieldsFunc(reply, func(r rune) bool {
		return r < '0' || r > '9'
	}) {
		v, err := strconv.Atoi(field)
		if err != nil || v < 1 || v > n {
			continue
		}
		if _, ok := seen[v-1]; ok {
			continue
		}
		seen[v-1] = struct{}{}
		out = append(out, v-1)
	}
	return out
}

func buildGradePrompt(question string, contexts []storage.Point) string {
	var b strings.Builder
	b.WriteString("Judge which documents contain information useful for answering the question.\n\n")
	b.WriteString("Documents:\n")
	for i, c := range contexts {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Text)
	}
	fmt.Fprintf(&b, "\nQuestion: %s\n", question)
	b.WriteString("\nReply with only the numbers of the useful documents, separated by commas. ")
	b.WriteString("If none of them are useful, reply with \"none\".")
	return b.String()
}

func buildRewritePrompt(question string) string {
	var b strings.Builder
	b.WriteString("The search results for this question were not useful.\n\n")
	fmt.Fprintf(&b, "Question: %s\n", question)
	b.WriteString("\nWrite a better search query for finding the documents that answer it. ")
	b.WriteString("Spell out the entities the question only refers to indirectly. ")
	b.WriteString("Reply with only the query.")
	return b.String()
}
