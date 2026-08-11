package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// VLLM to klient endpointów zgodnych z OpenAI (/v1/embeddings, /v1/chat/completions)
// serwowanych przez vLLM.
type VLLM struct {
	URL    string // np. http://localhost:8000
	Model  string
	Client *http.Client
}

// NewVLLM tworzy klienta VLLM z rozsądnymi wartościami domyślnymi.
func NewVLLM(url, model string) *VLLM {
	return &VLLM{URL: url, Model: model, Client: http.DefaultClient}
}

// Embed woła OpenAI-compatible endpoint /v1/embeddings.
func (v *VLLM) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body := map[string]any{
		"model": v.Model,
		"input": texts,
	}
	raw, err := doJSON(ctx, http.MethodPost, v.URL+"/v1/embeddings", body)
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

	out := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}

	return out, nil
}

// Generate woła OpenAI-compatible endpoint /v1/chat/completions.
func (v *VLLM) Generate(ctx context.Context, prompt string) (string, error) {
	body := map[string]any{
		"model": v.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	raw, err := doJSON(ctx, http.MethodPost, v.URL+"/v1/chat/completions", body)
	if err != nil {
		return "", err
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("decode chat completion response: %w", err)
	}

	if len(out.Choices) == 0 {
		return "", fmt.Errorf("llm returned no choices")
	}

	return out.Choices[0].Message.Content, nil
}

// doJSON to mały helper wysyłający JSON-owe żądanie HTTP i zwracający surową odpowiedź.
func doJSON(ctx context.Context, method, url string, in any) ([]byte, error) {
	body, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer res.Body.Close()

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if res.StatusCode >= 300 {
		return nil, fmt.Errorf("%s %s returned status %d: %s", method, url, res.StatusCode, string(raw))
	}

	return raw, nil
}
