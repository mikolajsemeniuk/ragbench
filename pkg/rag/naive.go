// Package rag zawiera implementacje baseline'owych architektur RAG
// (Retrieval-Augmented Generation) porównywanych w artykule.
package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Document to pojedynczy fragment korpusu, który trafia do bazy wektorowej.
type Document struct {
	ID   uint64
	Text string
}

// NaiveRAG to najprostszy, klasyczny pipeline RAG: embed -> retrieve top-K -> generate.
// Nie ma tu żadnej pętli, oceny jakości kontekstu ani decyzji agenta - to punkt
// odniesienia (baseline) dla bardziej zaawansowanych architektur (Self-RAG, CRAG, ...).
type NaiveRAG struct {
	QdrantURL  string // np. http://localhost:6333
	Collection string // nazwa kolekcji w Qdrant

	EmbedURL   string // np. http://localhost:8001 (vLLM, OpenAI-compatible /v1/embeddings)
	EmbedModel string // np. bge-base-en-v1.5

	LLMURL   string // np. http://localhost:8000 (vLLM, OpenAI-compatible /v1/chat/completions)
	LLMModel string // np. qwen2.5-7b-instruct

	TopK int // ile fragmentów kontekstu pobrać przed generacją

	Client *http.Client
}

// NewNaiveRAG tworzy NaiveRAG z rozsądnymi wartościami domyślnymi.
func NewNaiveRAG(qdrantURL, collection, embedURL, embedModel, llmURL, llmModel string) *NaiveRAG {
	return &NaiveRAG{
		QdrantURL:  qdrantURL,
		Collection: collection,
		EmbedURL:   embedURL,
		EmbedModel: embedModel,
		LLMURL:     llmURL,
		LLMModel:   llmModel,
		TopK:       5,
		Client:     http.DefaultClient,
	}
}

// EnsureCollection tworzy kolekcję w Qdrant, jeśli jeszcze nie istnieje.
// vectorSize musi odpowiadać wymiarowi wektorów zwracanych przez model embeddingowy
// (np. 768 dla BAAI/bge-base-en-v1.5).
func (r *NaiveRAG) EnsureCollection(ctx context.Context, vectorSize int) error {
	body := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": "Cosine",
		},
	}
	_, err := r.doJSON(ctx, http.MethodPut, r.QdrantURL+"/collections/"+r.Collection, body)
	return err
}

// Ingest liczy embeddingi dla dokumentów i zapisuje je (wraz z tekstem) do Qdrant.
func (r *NaiveRAG) Ingest(ctx context.Context, docs []Document) error {
	if len(docs) == 0 {
		return nil
	}

	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.Text
	}

	vectors, err := r.embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed documents: %w", err)
	}

	points := make([]map[string]any, len(docs))
	for i, d := range docs {
		points[i] = map[string]any{
			"id":     d.ID,
			"vector": vectors[i],
			"payload": map[string]any{
				"text": d.Text,
			},
		}
	}

	body := map[string]any{"points": points}
	_, err = r.doJSON(ctx, http.MethodPut, r.QdrantURL+"/collections/"+r.Collection+"/points?wait=true", body)
	if err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}
	return nil
}

// Query to główna metoda baseline'u: dla pytania użytkownika pobiera TopK
// najbardziej podobnych fragmentów tekstu z Qdrant, wkleja je do promptu
// i zwraca odpowiedź wygenerowaną przez LLM.
func (r *NaiveRAG) Query(ctx context.Context, question string) (string, error) {
	contexts, err := r.retrieve(ctx, question)
	if err != nil {
		return "", fmt.Errorf("retrieve: %w", err)
	}

	answer, err := r.generate(ctx, question, contexts)
	if err != nil {
		return "", fmt.Errorf("generate: %w", err)
	}
	return answer, nil
}

// retrieve embeduje pytanie i szuka TopK najbliższych fragmentów w Qdrant.
func (r *NaiveRAG) retrieve(ctx context.Context, question string) ([]string, error) {
	vectors, err := r.embed(ctx, []string{question})
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}

	body := map[string]any{
		"vector":       vectors[0],
		"limit":        r.TopK,
		"with_payload": true,
	}
	raw, err := r.doJSON(ctx, http.MethodPost, r.QdrantURL+"/collections/"+r.Collection+"/points/search", body)
	if err != nil {
		return nil, fmt.Errorf("search qdrant: %w", err)
	}

	var resp struct {
		Result []struct {
			Payload struct {
				Text string `json:"text"`
			} `json:"payload"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	contexts := make([]string, len(resp.Result))
	for i, res := range resp.Result {
		contexts[i] = res.Payload.Text
	}
	return contexts, nil
}

// generate wysyła pytanie wraz z pobranym kontekstem do LLM (vLLM, endpoint
// zgodny z OpenAI) i zwraca wygenerowaną odpowiedź.
func (r *NaiveRAG) generate(ctx context.Context, question string, contexts []string) (string, error) {
	prompt := buildPrompt(question, contexts)

	body := map[string]any{
		"model": r.LLMModel,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	raw, err := r.doJSON(ctx, http.MethodPost, r.LLMURL+"/v1/chat/completions", body)
	if err != nil {
		return "", fmt.Errorf("call llm: %w", err)
	}

	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("decode llm response: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("llm returned no choices")
	}
	return resp.Choices[0].Message.Content, nil
}

// buildPrompt składa prosty prompt: kontekst z bazy wektorowej + pytanie.
// To jest "Augment" z pipeline'u retrieve -> augment -> generate.
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

// embed woła OpenAI-compatible endpoint /v1/embeddings serwowany przez vLLM.
func (r *NaiveRAG) embed(ctx context.Context, texts []string) ([][]float32, error) {
	body := map[string]any{
		"model": r.EmbedModel,
		"input": texts,
	}
	raw, err := r.doJSON(ctx, http.MethodPost, r.EmbedURL+"/v1/embeddings", body)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("decode embeddings response: %w", err)
	}

	vectors := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		vectors[i] = d.Embedding
	}
	return vectors, nil
}

// doJSON to mały helper wysyłający JSON-owe żądanie HTTP i zwracający surową odpowiedź.
func (r *NaiveRAG) doJSON(ctx context.Context, method, url string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%s %s returned status %d: %s", method, url, resp.StatusCode, string(raw))
	}
	return raw, nil
}
