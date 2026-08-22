package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Ollama is a client for a local Ollama server (/api/embeddings, /api/generate).
//
// Unlike vLLM, Ollama exposes no way to cap the prompt at a given number of
// tokens, so callers that need a hard length bound must approximate it
// themselves (see cmd/ingest, -max-words).
type Ollama struct {
	URL    string // e.g. http://localhost:11434
	Model  string
	Client *http.Client
}

// NewOllama creates an Ollama client with sensible defaults.
func NewOllama(url, model string) *Ollama {
	return &Ollama{URL: url, Model: model, Client: NewHTTPClient(5*time.Minute, 64)}
}

// Embed calls /api/embeddings once per text, because that endpoint takes a
// single prompt and has no batch form.
func (o *Ollama) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		body := map[string]any{
			"model":  o.Model,
			"prompt": text,
		}
		raw, err := doJSON(ctx, o.Client, http.MethodPost, o.URL+"/api/embeddings", body)
		if err != nil {
			return nil, err
		}

		var resp struct {
			Embedding []float32 `json:"embedding"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			return nil, fmt.Errorf("decode embedding response: %w", err)
		}
		if len(resp.Embedding) == 0 {
			return nil, fmt.Errorf("embedding response for input %d is empty", i)
		}
		vectors[i] = resp.Embedding
	}
	return vectors, nil
}

// Generate calls /api/generate with streaming disabled.
func (o *Ollama) Generate(ctx context.Context, prompt string) (string, error) {
	body := map[string]any{
		"model":  o.Model,
		"prompt": prompt,
		"stream": false,
	}
	raw, err := doJSON(ctx, o.Client, http.MethodPost, o.URL+"/api/generate", body)
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
