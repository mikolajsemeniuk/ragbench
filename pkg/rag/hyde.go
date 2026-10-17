package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// HyDE searches with a fabricated answer instead of the question (Gao et al.,
// https://arxiv.org/abs/2212.10496).
//
// A dense retriever compares the question's vector with each passage's vector,
// but a question and the passage that answers it are different kinds of text.
// "Where was the place of death of the director of film Beat Girl?" shares
// almost no vocabulary with an encyclopedia paragraph about Edmond T.
// Greville, so the two vectors are far apart however good the encoder is.
//
// HyDE removes that mismatch by asking the generator to write the passage it
// thinks would answer the question, and searching with that instead. The
// draft may be factually wrong - it usually is in the details - and it does
// not matter: it only has to look like the right passage, so that the encoder
// puts it near the real one. The wrong details are discarded; only the vector
// is used.
//
// This targets the failure bucket cmd/diagnose calls "not reachable by this
// query": 53.3% of retrieval failures on MuSiQue, 24.7% on 2WikiMultihopQA,
// 23.0% on HotpotQA.
//
// Cost is one extra generation call per question, which makes it the cheapest
// of the query-side methods compared here - IRCoT spends between two and six.
type HyDE struct {
	Store      storage.VectorStore
	Collection string

	Embedder  Embedder
	Generator Generator

	// Drafter writes the hypothetical passage. It is a separate Generator
	// because it needs a different token budget: the answer is a short span
	// and is capped accordingly, while a draft cut off after 64 tokens is half
	// a sentence and embeds poorly.
	Drafter Generator

	TopK int

	// IncludeQuestion averages the question's own embedding into the search
	// vector, as the paper does. It is the safety net for the case the draft
	// goes off topic: the search then degrades towards plain NaiveRAG instead
	// of chasing a hallucination.
	IncludeQuestion bool

	QueryPrefix    string
	DocumentPrefix string
}

func NewHyDE(store storage.VectorStore, collection string, embedder Embedder, generator, drafter Generator) *HyDE {
	return &HyDE{
		Store:           store,
		Collection:      collection,
		Embedder:        embedder,
		Generator:       generator,
		Drafter:         drafter,
		TopK:            5,
		IncludeQuestion: true,
	}
}

func (r *HyDE) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	draft, err := r.Drafter.Generate(ctx, buildHyDEPrompt(question))
	if err != nil {
		return "", nil, fmt.Errorf("draft hypothetical passage: %w", err)
	}
	draft = strings.TrimSpace(draft)

	// The draft is encoded as a document and the question as a query, because
	// that is what each of them is. For an encoder with asymmetric
	// instructions (E5, Nomic Embed) using the wrong side would silently cost
	// retrieval quality; for BGE English v1.5 both prefixes are empty and this
	// is a no-op.
	var texts []string
	if draft != "" {
		texts = append(texts, r.DocumentPrefix+draft)
	}
	if draft == "" || r.IncludeQuestion {
		texts = append(texts, r.QueryPrefix+question)
	}

	vectors, err := r.Embedder.Embed(ctx, texts)
	if err != nil {
		return "", nil, fmt.Errorf("embed hypothetical passage: %w", err)
	}

	points, err := r.Store.Search(ctx, r.Collection, meanVector(vectors), r.TopK)
	if err != nil {
		return "", nil, fmt.Errorf("search: %w", err)
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, points))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, points, nil
}

// meanVector averages the embeddings component-wise. The result is not
// re-normalised: the collection uses cosine distance, and Qdrant normalises
// the query vector itself, so only the direction of the average matters.
func meanVector(vectors [][]float32) []float32 {
	if len(vectors) == 1 {
		return vectors[0]
	}

	out := make([]float32, len(vectors[0]))
	for _, v := range vectors {
		for i := range out {
			out[i] += v[i]
		}
	}
	n := float32(len(vectors))
	for i := range out {
		out[i] /= n
	}
	return out
}

// buildHyDEPrompt asks for a passage, not an answer. Naming entities
// explicitly is what makes the draft useful: the vocabulary the question is
// missing ("Edmond T. Greville", "Nice") is exactly the vocabulary that has to
// end up in the search vector.
func buildHyDEPrompt(question string) string {
	var b strings.Builder
	b.WriteString("Write a short encyclopedia passage that answers the question.\n\n")
	fmt.Fprintf(&b, "Question: %s\n", question)
	b.WriteString("\nName the people, places, dates and titles involved. ")
	b.WriteString("Write it as an encyclopedia entry, not as a reply to the question. ")
	b.WriteString("If you are not sure of the facts, write the most plausible passage anyway. ")
	b.WriteString("Reply with only the passage.")
	return b.String()
}
