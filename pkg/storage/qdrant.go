package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type Qdrant struct {
	URL    string
	Client *http.Client
}

func NewQdrant(url string) *Qdrant {
	return &Qdrant{URL: url, Client: http.DefaultClient}
}

func (q *Qdrant) EnsureCollection(ctx context.Context, collection string, vectorSize int) error {
	body := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": "Cosine",
		},
	}
	_, err := q.doJSON(ctx, http.MethodPut, q.URL+"/collections/"+collection, body)
	return err
}

func (q *Qdrant) Upsert(ctx context.Context, collection string, points []Point) error {
	if len(points) == 0 {
		return nil
	}

	payload := make([]map[string]any, len(points))
	for i, p := range points {
		payload[i] = map[string]any{
			"id":     p.ID,
			"vector": p.Vector,
			"payload": map[string]any{
				"text": p.Text,
			},
		}
	}

	body := map[string]any{"points": payload}
	_, err := q.doJSON(ctx, http.MethodPut, q.URL+"/collections/"+collection+"/points?wait=true", body)
	if err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}
	return nil
}

func (q *Qdrant) Search(ctx context.Context, collection string, vector []float32, limit int) ([]string, error) {
	body := map[string]any{
		"vector":       vector,
		"limit":        limit,
		"with_payload": true,
	}
	raw, err := q.doJSON(ctx, http.MethodPost, q.URL+"/collections/"+collection+"/points/search", body)
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

	texts := make([]string, len(resp.Result))
	for i, res := range resp.Result {
		texts[i] = res.Payload.Text
	}
	return texts, nil
}

func (q *Qdrant) doJSON(ctx context.Context, method, url string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := q.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
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
