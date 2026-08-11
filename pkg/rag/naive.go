// Package rag zawiera implementacje baseline'owych architektur RAG
// (Retrieval-Augmented Generation) porównywanych w artykule.
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
		texts[i] = d.Text
	}

	vectors, err := r.Embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed documents: %w", err)
	}

	points := make([]storage.Point, len(docs))
	for i, d := range docs {
		points[i] = storage.Point{ID: d.ID, Vector: vectors[i], Text: d.Text}
	}

	if err := r.Store.Upsert(ctx, r.Collection, points); err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}

	return nil
}

func (r *NaiveRAG) Query(ctx context.Context, question string) (string, error) {
	contexts, err := r.retrieve(ctx, question)
	if err != nil {
		return "", fmt.Errorf("retrieve: %w", err)
	}

	answer, err := r.Generator.Generate(ctx, buildPrompt(question, contexts))
	if err != nil {
		return "", fmt.Errorf("generate: %w", err)
	}
	return answer, nil
}

func (r *NaiveRAG) retrieve(ctx context.Context, question string) ([]string, error) {
	vectors, err := r.Embedder.Embed(ctx, []string{question})
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}

	contexts, err := r.Store.Search(ctx, r.Collection, vectors[0], r.TopK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	return contexts, nil
}

func buildPrompt(question string, contexts []string) string {
	var b bytes.Buffer
	b.WriteString("Odpowiedz na pytanie wyłącznie na podstawie poniższego kontekstu. ")
	b.WriteString("Jeśli kontekst nie zawiera odpowiedzi, powiedz, że nie wiesz.\n\n")
	b.WriteString("Kontekst:\n")
	for i, c := range contexts {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c)
	}
	b.WriteString("\nPytanie: ")
	b.WriteString(question)
	return b.String()
}
