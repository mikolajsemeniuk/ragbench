package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// Ollama to klient lokalnego serwera Ollama (/api/embeddings, /api/generate).
type Ollama struct {
	URL    string // np. http://localhost:11434
	Model  string
	Client *http.Client
}

// NewOllama tworzy klienta Ollama z rozsądnymi wartościami domyślnymi.
func NewOllama(url, model string) *Ollama {
	return &Ollama{URL: url, Model: model, Client: http.DefaultClient}
}

// Embed woła endpoint /api/embeddings dla każdego tekstu z osobna,
// ponieważ Ollama nie wspiera batchowania w tym endpoincie.
func (o *Ollama) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		body := map[string]any{
			"model":  o.Model,
			"prompt": text,
		}
		raw, err := doJSON(ctx, http.MethodPost, o.URL+"/api/embeddings", body)
		if err != nil {
			return nil, err
		}

		var resp struct {
			Embedding []float32 `json:"embedding"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			return nil, fmt.Errorf("decode embedding response: %w", err)
		}
		vectors[i] = resp.Embedding
	}
	return vectors, nil
}

// Generate woła endpoint /api/generate z wyłączonym strumieniowaniem.
func (o *Ollama) Generate(ctx context.Context, prompt string) (string, error) {
	body := map[string]any{
		"model":  o.Model,
		"prompt": prompt,
		"stream": false,
	}
	raw, err := doJSON(ctx, http.MethodPost, o.URL+"/api/generate", body)
	if err != nil {
		return "", err
	}

	var resp struct {
		Response string `json:"response"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("decode generate response: %w", err)
	}
	return resp.Response, nil
}
