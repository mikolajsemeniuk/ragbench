package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// IRCoT interleaves retrieval with chain-of-thought reasoning (Trivedi et al.,
// https://arxiv.org/abs/2212.10509).
//
// NaiveRAG issues one query, built from the question alone. On a multi-hop
// question that is structurally insufficient: "Were Scott Derrickson and Ed
// Wood of the same nationality?" names both entities, but "What is the capital
// of the country where X was born?" never names the country, so no single
// query can reach the passage that answers it. The measured effect on the
// NaiveRAG baseline is Recall@5 falling from 0.51 on HotpotQA to 0.30 on
// 2WikiMultihopQA to 0.20 on MuSiQue as the number of required hops grows.
//
// IRCoT closes that gap by alternating: reason one sentence, use that sentence
// as the next query, add what comes back, reason again. The reasoning step
// names entities the question never mentioned, which is what makes the second
// hop retrievable.
//
// The answer is produced by the same reader prompt NaiveRAG uses, over the
// accumulated passages. That is deliberate: it keeps the *only* difference
// between the two systems the set of passages in context, so a difference in
// scores measures retrieval, not answer formatting. It also matches the
// "IRCoT QA" configuration of the paper, which pairs IRCoT retrieval with a
// separate reader.
type IRCoT struct {
	Store      storage.VectorStore
	Collection string

	Embedder  Embedder
	Generator Generator

	// Stepper produces the reasoning sentences. It is separate from Generator
	// because the two calls want opposite token budgets: the final answer is a
	// short span and is capped hard, while a reasoning sentence cut off at the
	// same cap is half a sentence - and that half sentence is then used
	// verbatim as the next retrieval query. Nil falls back to Generator.
	Stepper Generator

	// TopK is the number of passages fetched per retrieval step, and the K
	// reported as Recall@K. MaxSteps bounds the reasoning loop, and
	// MaxPassages bounds the accumulated context so it cannot outgrow the
	// generator's window on questions that keep retrieving.
	TopK        int
	MaxSteps    int
	MaxPassages int

	QueryPrefix    string
	DocumentPrefix string
}

func NewIRCoT(store storage.VectorStore, collection string, embedder Embedder, generator Generator) *IRCoT {
	return &IRCoT{
		Store:       store,
		Collection:  collection,
		Embedder:    embedder,
		Generator:   generator,
		TopK:        5,
		MaxSteps:    4,
		MaxPassages: 15,
	}
}

// Query runs the interleaved loop and returns the answer together with the
// passages that ended up in context.
//
// Note what is returned for scoring: the accumulated passage set, capped at
// MaxPassages, not the top TopK of a single query. Recall@K for IRCoT is
// therefore recall over everything the loop gathered - which is the quantity
// the method is trying to improve.
func (r *IRCoT) Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error) {
	seen := make(map[uint64]struct{})
	var passages []storage.Point

	add := func(points []storage.Point) {
		for _, p := range points {
			if _, ok := seen[p.ID]; ok {
				continue
			}
			if len(passages) >= r.MaxPassages {
				return
			}
			seen[p.ID] = struct{}{}
			passages = append(passages, p)
		}
	}

	first, err := r.retrieve(ctx, question)
	if err != nil {
		return "", nil, fmt.Errorf("initial retrieve: %w", err)
	}
	add(first)

	var reasoning []string
	for range r.MaxSteps {
		if len(passages) >= r.MaxPassages {
			break
		}

		sentence, err := r.stepper().Generate(ctx, buildReasoningPrompt(question, passages, reasoning))
		if err != nil {
			return "", nil, fmt.Errorf("reasoning step: %w", err)
		}
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			break
		}
		reasoning = append(reasoning, sentence)
		TraceFrom(ctx).AddReasoning(sentence)

		// The model signalling it can answer means further retrieval has
		// nothing left to contribute.
		if strings.Contains(strings.ToLower(sentence), "so the answer is") {
			break
		}

		next, err := r.retrieve(ctx, sentence)
		if err != nil {
			return "", nil, fmt.Errorf("retrieve for reasoning step: %w", err)
		}
		before := len(passages)
		add(next)
		// A step that surfaced nothing new means the loop has converged;
		// continuing would burn a generation call per question for nothing.
		if len(passages) == before {
			break
		}
	}

	answer, err = r.Generator.Generate(ctx, buildPrompt(question, passages))
	if err != nil {
		return "", nil, fmt.Errorf("generate: %w", err)
	}
	return answer, passages, nil
}

// stepper returns the generator to use for an intermediate call.
func (r *IRCoT) stepper() Generator {
	if r.Stepper != nil {
		return r.Stepper
	}
	return r.Generator
}

func (r *IRCoT) retrieve(ctx context.Context, query string) ([]storage.Point, error) {
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

// buildReasoningPrompt asks for one sentence at a time. One sentence, rather
// than a full chain, is what makes the loop work: each sentence becomes the
// next retrieval query, so a long chain would blur several hops into a single
// query and lose the very specificity that lets the second hop be found.
func buildReasoningPrompt(question string, contexts []storage.Point, reasoning []string) string {
	var b strings.Builder
	b.WriteString("You are answering a question step by step using the documents below.\n\n")
	b.WriteString("Documents:\n")
	for i, c := range contexts {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, c.Text)
	}
	fmt.Fprintf(&b, "\nQuestion: %s\n", question)
	if len(reasoning) > 0 {
		fmt.Fprintf(&b, "\nReasoning so far:\n%s\n", strings.Join(reasoning, "\n"))
	}
	b.WriteString("\nWrite the next single sentence of reasoning. ")
	b.WriteString("Name the entities that still need to be looked up. ")
	b.WriteString("If the documents already answer the question, write \"So the answer is: <answer>\".")
	return b.String()
}
