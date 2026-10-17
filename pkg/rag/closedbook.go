package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// ClosedBook answers from the generator's own parametric knowledge, with no
// retrieval at all.
//
// This is the floor every retrieval-augmented number needs to be read against.
// A 7B instruct model has memorised a large amount of Wikipedia-scale trivia,
// so a score like 0.568 Exact Match on TriviaQA does not by itself show that
// retrieval contributed anything - the model may simply know the answers. The
// difference between this baseline and NaiveRAG is what retrieval is actually
// worth on each dataset.
//
// It returns no passages, so answer-in-context and Recall@K come out at 0 by
// construction. That is correct rather than a gap: nothing was placed in
// context, so nothing relevant could be.
type ClosedBook struct {
	Generator Generator
}

func NewClosedBook(generator Generator) *ClosedBook {
	return &ClosedBook{Generator: generator}
}

func (r *ClosedBook) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	answer, err = r.Generator.Generate(ctx, buildClosedBookPrompt(question))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, nil, nil
}

// buildClosedBookPrompt mirrors buildPrompt minus the documents, so that the
// only difference from NaiveRAG is the presence of retrieved context and not
// the way the answer is requested.
func buildClosedBookPrompt(question string) string {
	var b strings.Builder
	b.WriteString("Answer the question. ")
	b.WriteString("Only give me the answer and do not output any other words.\n\n")
	b.WriteString("Question: ")
	b.WriteString(question)
	return b.String()
}
