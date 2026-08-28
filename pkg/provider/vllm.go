package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// VLLM is a client for the OpenAI-compatible endpoints (/v1/embeddings,
// /v1/chat/completions) served by vLLM.
type VLLM struct {
	URL    string // e.g. http://localhost:8001
	Model  string
	Client *http.Client

	// TruncatePromptTokens, when > 0, is sent as truncate_prompt_tokens with
	// every embedding request, telling vLLM to truncate inputs longer than
	// this to exactly this many tokens instead of rejecting them with 400
	// "maximum context length".
	//
	// Truncating on the server is what makes the ingest deterministic. The
	// alternative - truncating client-side via /tokenize + /detokenize - is
	// not token-exact: decoding a token prefix back to text and re-encoding
	// it can yield MORE tokens than the prefix had (unknown characters
	// round-trip as the literal "[UNK]", casing/accent normalisation and
	// WordPiece re-splitting shift boundaries), so documents trimmed to the
	// limit still came back over it and the whole batch was rejected. Here
	// the model's own tokenizer does the cut, on the exact token sequence it
	// is about to embed, so the limit cannot be overshot.
	TruncatePromptTokens int

	// Temperature and MaxTokens pin down generation. Left unset, vLLM samples
	// with temperature 1.0, so the same question yields a different answer on
	// every run and no reported number can be reproduced - by a reviewer or by
	// us. Short-form QA wants greedy decoding (temperature 0) and a hard cap
	// on the answer length.
	Temperature float64
	MaxTokens   int
}

// NewVLLM creates a VLLM client with defaults suited to a long ingest run.
func NewVLLM(url, model string) *VLLM {
	return &VLLM{
		URL:         url,
		Model:       model,
		Client:      NewHTTPClient(5*time.Minute, 64),
		Temperature: 0,
		MaxTokens:   64,
	}
}

// Health reports whether the server is up and serving. Used to fail fast with
// a clear message instead of burning the whole retry budget at startup.
func (v *VLLM) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.URL+"/health", nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	client := v.Client
	if client == nil {
		client = http.DefaultClient
	}
	res, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 300 {
		return &HTTPError{Method: http.MethodGet, URL: v.URL + "/health", Status: res.StatusCode}
	}
	return nil
}

// Embed calls the OpenAI-compatible /v1/embeddings endpoint.
//
// The returned slice is aligned with texts by the response's index field
// rather than by arrival order. The OpenAI schema does not promise that data
// comes back in request order, and a silent misalignment here would attach
// every vector to the wrong passage - producing a corpus that looks perfectly
// healthy (right number of points, right dimensions) while retrieval is
// nonsense. The count and dimension checks below exist for the same reason:
// a partial response must fail loudly, not shift the remaining vectors by one.
func (v *VLLM) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body := map[string]any{
		"model":           v.Model,
		"input":           texts,
		"encoding_format": "float",
	}
	if v.TruncatePromptTokens > 0 {
		body["truncate_prompt_tokens"] = v.TruncatePromptTokens
	}

	raw, err := doJSON(ctx, v.Client, http.MethodPost, v.URL+"/v1/embeddings", body)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("decode embeddings response: %w", err)
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings response has %d vectors for %d inputs", len(resp.Data), len(texts))
	}

	out := make([][]float32, len(texts))
	for _, d := range resp.Data {
		if d.Index < 0 || d.Index >= len(texts) {
			return nil, fmt.Errorf("embeddings response has out-of-range index %d for %d inputs", d.Index, len(texts))
		}
		if out[d.Index] != nil {
			return nil, fmt.Errorf("embeddings response has duplicate index %d", d.Index)
		}
		if len(d.Embedding) == 0 {
			return nil, fmt.Errorf("embeddings response has empty vector at index %d", d.Index)
		}
		out[d.Index] = d.Embedding
	}
	for i, vec := range out {
		if vec == nil {
			return nil, fmt.Errorf("embeddings response is missing index %d", i)
		}
		if len(vec) != len(out[0]) {
			return nil, fmt.Errorf("embeddings response mixes vector dimensions: %d at index 0, %d at index %d", len(out[0]), len(vec), i)
		}
	}
	return out, nil
}

// Rerank scores every document against the query with a cross-encoder and
// returns the indices of the top n, best first.
//
// A bi-encoder - what the retriever uses - embeds the question and the passage
// separately and compares the two vectors, so it never sees them together. A
// cross-encoder reads the pair jointly and can therefore judge relevance far
// more precisely. It is too slow to run over a whole corpus, which is exactly
// why it is used as a second stage over a shortlist the retriever produced.
func (v *VLLM) Rerank(ctx context.Context, query string, documents []string, n int) ([]int, error) {
	if len(documents) == 0 {
		return nil, nil
	}
	if n > len(documents) {
		n = len(documents)
	}

	body := map[string]any{
		"model":     v.Model,
		"query":     query,
		"documents": documents,
		"top_n":     n,
	}
	raw, err := doJSON(ctx, v.Client, http.MethodPost, v.URL+"/v1/rerank", body)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Results []struct {
			Index int `json:"index"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("decode rerank response: %w", err)
	}
	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("rerank response is empty for %d documents", len(documents))
	}

	out := make([]int, 0, len(resp.Results))
	for _, r := range resp.Results {
		if r.Index < 0 || r.Index >= len(documents) {
			return nil, fmt.Errorf("rerank response has out-of-range index %d for %d documents", r.Index, len(documents))
		}
		out = append(out, r.Index)
	}
	return out, nil
}

// Generate calls the OpenAI-compatible /v1/chat/completions endpoint.
func (v *VLLM) Generate(ctx context.Context, prompt string) (string, error) {
	body := map[string]any{
		"model": v.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": v.Temperature,
	}
	if v.MaxTokens > 0 {
		body["max_tokens"] = v.MaxTokens
	}
	raw, err := doJSON(ctx, v.Client, http.MethodPost, v.URL+"/v1/chat/completions", body)
	if err != nil {
		return "", err
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("decode chat completion response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("llm returned no choices")
	}
	recordUsage(ctx, out.Usage.PromptTokens, out.Usage.CompletionTokens)
	return out.Choices[0].Message.Content, nil
}
