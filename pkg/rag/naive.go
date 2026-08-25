// Package rag contains the baseline Retrieval-Augmented Generation
// architectures compared in the paper.
package rag

import (
	"bytes"
	"context"
	"fmt"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type Generator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

type Document struct {
	ID   uint64
	Text string
}

type NaiveRAG struct {
	Store      storage.VectorStore
	Collection string

	Embedder  Embedder
	Generator Generator

	TopK int

	// QueryPrefix and DocumentPrefix are the asymmetric instructions the
	// embedding model expects - see PrefixesFor. They are applied to the text
	// handed to the encoder only: the passage stored in the vector store, and
	// therefore the context handed to the generator, is always the clean
	// original.
	QueryPrefix    string
	DocumentPrefix string
}

func NewNaiveRAG(store storage.VectorStore, collection string, embedder Embedder, generator Generator) *NaiveRAG {
	return &NaiveRAG{
		Store:      store,
		Collection: collection,
		Embedder:   embedder,
		Generator:  generator,
		TopK:       5,
	}
}

func (r *NaiveRAG) EnsureCollection(ctx context.Context, vectorSize int) error {
	return r.Store.EnsureCollection(ctx, r.Collection, vectorSize)
}

func (r *NaiveRAG) Ingest(ctx context.Context, docs []Document) error {
	if len(docs) == 0 {
		return nil
	}

	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = r.DocumentPrefix + d.Text
	}

	vectors, err := r.Embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed documents: %w", err)
	}

	points := make([]storage.Point, len(docs))
	for i, d := range docs {
		// d.Text, not the prefixed text: the prefix is an instruction to the
		// encoder, not part of the passage.
		points[i] = storage.Point{ID: d.ID, Vector: vectors[i], Text: d.Text}
	}

	if err := r.Store.Upsert(ctx, r.Collection, points); err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}

	return nil
}

// Query runs the full NaiveRAG pass and returns the model's answer together
// with the retrieved passages, ordered by decreasing relevance. The passages
// themselves are returned - not just their IDs - because cmd/bench derives
// every retrieval metric from them: the article title is the first line of a
// FlashRAG passage, and answer-in-context recall needs the text.
func (r *NaiveRAG) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	points, err := r.retrieve(ctx, question)
	if err != nil {
		return "", nil, fmt.Errorf("retrieve: %w", err)
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, points))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, points, nil
}

func (r *NaiveRAG) retrieve(ctx context.Context, question string) ([]storage.Point, error) {
	vectors, err := r.Embedder.Embed(ctx, []string{r.QueryPrefix + question})
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}

	points, err := r.Store.Search(ctx, r.Collection, vectors[0], r.TopK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	return points, nil
}

// buildPrompt renders the baseline prompt.
//
// The instruction to answer with nothing but the answer is not stylistic. The
// gold answers in these datasets are short spans ("yes", "1994", "Tokyo"),
// and Exact Match compares against them literally. An instruct model left to
// answer freely replies "Yes, Scott Derrickson and Ed Wood were both
// American", which scores EM 0 while being entirely correct. Measured over 50
// HotpotQA dev questions, constraining the output moves EM from 0.02 to 0.40
// and token-level F1 from 0.11 to 0.53 - without touching retrieval. The
// phrasing follows the convention used by FlashRAG and the works it compares.
//
// It is in English on purpose: the
// corpora (FlashRAG wiki18_100w) and the question sets (NQ, TriviaQA,
// HotpotQA, 2WikiMultihopQA, MuSiQue) are English, the gold answers are
// English, and Exact Match / token-level F1 are computed against them - a
// prompt in another language would push the model to answer in that language
// and depress both metrics for reasons that have nothing to do with retrieval.
func buildPrompt(question string, contexts []storage.Point) string {
	var b bytes.Buffer
	b.WriteString("Answer the question based on the given documents. ")
	b.WriteString("Only give me the answer and do not output any other words.\n\n")
	b.WriteString("Documents:\n")
	for i, c := range contexts {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Text)
	}
	b.WriteString("\nQuestion: ")
	b.WriteString(question)
	return b.String()
}
