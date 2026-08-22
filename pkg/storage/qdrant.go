package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
)

// CollectionConfig describes how a collection should be created. The zero
// value is a plain in-memory collection; DefaultCollectionConfig returns the
// configuration used for the full-corpus ingest.
type CollectionConfig struct {
	VectorSize int
	Distance   string // Cosine | Dot | Euclid

	// OnDiskVectors and OnDiskPayload keep vectors and passage text in
	// memory-mapped files instead of RAM. The full FlashRAG wiki18_100w
	// corpus is ~21M passages: at 768 float32 dimensions that is ~64 GiB of
	// raw vectors plus ~13 GiB of text, which does not fit in RAM next to a
	// GPU serving stack on a single host. On disk, the OS page cache keeps
	// the hot part resident and the collection simply works.
	OnDiskVectors bool
	OnDiskPayload bool
	OnDiskHNSW    bool

	// IndexingThreshold is the number of vectors a segment must hold before
	// Qdrant builds an HNSW index for it. Setting it to 0 during a bulk load
	// disables index construction entirely, so the ingest is not competing
	// with the indexer for CPU and does not build indexes it will discard as
	// segments keep growing. It is restored afterwards - see
	// SetIndexingThreshold - which triggers one final indexing pass.
	IndexingThreshold *int
}

// DefaultCollectionConfig returns the bulk-load configuration for a corpus of
// the given vector size.
func DefaultCollectionConfig(vectorSize int) CollectionConfig {
	zero := 0
	return CollectionConfig{
		VectorSize:        vectorSize,
		Distance:          "Cosine",
		OnDiskVectors:     true,
		OnDiskPayload:     true,
		OnDiskHNSW:        true,
		IndexingThreshold: &zero,
	}
}

type Qdrant struct {
	URL    string
	Client *http.Client

	// Wait makes every upsert block until the points are durably applied.
	// This is what turns "the request returned 200" into "the points are in
	// the collection", and it is also the backpressure that stops the ingest
	// from queueing more work than Qdrant can absorb.
	Wait bool
}

func NewQdrant(url string) *Qdrant {
	return &Qdrant{URL: url, Client: provider.NewHTTPClient(5*time.Minute, 64), Wait: true}
}

// EnsureCollection creates the collection with the default bulk-load
// configuration if it does not exist yet.
func (q *Qdrant) EnsureCollection(ctx context.Context, collection string, vectorSize int) error {
	return q.EnsureCollectionWithConfig(ctx, collection, DefaultCollectionConfig(vectorSize))
}

// EnsureCollectionWithConfig creates the collection if it does not exist yet.
// If it already exists its vector size is verified, so a run against a
// collection built with a different embedding model fails immediately instead
// of silently mixing incompatible vectors into one index.
func (q *Qdrant) EnsureCollectionWithConfig(ctx context.Context, collection string, cfg CollectionConfig) error {
	info, err := q.collectionInfo(ctx, collection)
	if err != nil {
		return fmt.Errorf("check collection exists: %w", err)
	}
	if info != nil {
		if got := info.Result.Config.Params.Vectors.Size; got != 0 && got != cfg.VectorSize {
			return fmt.Errorf("collection %q already exists with vector size %d, but the embedding model produces %d - use a different -collection or delete the existing one", collection, got, cfg.VectorSize)
		}
		return nil
	}

	distance := cfg.Distance
	if distance == "" {
		distance = "Cosine"
	}

	body := map[string]any{
		"vectors": map[string]any{
			"size":     cfg.VectorSize,
			"distance": distance,
			"on_disk":  cfg.OnDiskVectors,
		},
		"on_disk_payload": cfg.OnDiskPayload,
		"hnsw_config": map[string]any{
			"on_disk": cfg.OnDiskHNSW,
		},
	}
	if cfg.IndexingThreshold != nil {
		body["optimizers_config"] = map[string]any{"indexing_threshold": *cfg.IndexingThreshold}
	}

	_, err = q.doJSON(ctx, http.MethodPut, q.URL+"/collections/"+collection, body)
	return err
}

// SetIndexingThreshold updates the collection's indexing threshold. Called
// after a bulk load with the production value to let Qdrant build the HNSW
// index once, over finished segments.
func (q *Qdrant) SetIndexingThreshold(ctx context.Context, collection string, threshold int) error {
	body := map[string]any{
		"optimizers_config": map[string]any{"indexing_threshold": threshold},
	}
	_, err := q.doJSON(ctx, http.MethodPatch, q.URL+"/collections/"+collection, body)
	return err
}

// CountPoints returns the exact number of points stored in the collection.
// Used to verify an ingest against the number of documents in the corpus
// rather than trusting the ingest's own bookkeeping.
func (q *Qdrant) CountPoints(ctx context.Context, collection string) (int64, error) {
	raw, err := q.doJSON(ctx, http.MethodPost, q.URL+"/collections/"+collection+"/points/count", map[string]any{"exact": true})
	if err != nil {
		return 0, err
	}

	var out struct {
		Result struct {
			Count int64 `json:"count"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, fmt.Errorf("decode count response: %w", err)
	}

	return out.Result.Count, nil
}

type collectionInfoResponse struct {
	Result struct {
		Config struct {
			Params struct {
				Vectors struct {
					Size int `json:"size"`
				} `json:"vectors"`
			} `json:"params"`
		} `json:"config"`
	} `json:"result"`
}

// collectionInfo returns nil, nil when the collection does not exist.
func (q *Qdrant) collectionInfo(ctx context.Context, collection string) (*collectionInfoResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, q.URL+"/collections/"+collection, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

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
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode >= 300 {
		return nil, &provider.HTTPError{Method: http.MethodGet, URL: q.URL + "/collections/" + collection, Status: resp.StatusCode, Body: string(raw)}
	}

	var info collectionInfoResponse
	if err := json.Unmarshal(raw, &info); err != nil {
		return nil, fmt.Errorf("decode collection info: %w", err)
	}
	return &info, nil
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

	url := q.URL + "/collections/" + collection + "/points"
	if q.Wait {
		url += "?wait=true"
	}
	if _, err := q.doJSON(ctx, http.MethodPut, url, map[string]any{"points": payload}); err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}
	return nil
}

func (q *Qdrant) Search(ctx context.Context, collection string, vector []float32, limit int) ([]Point, error) {
	body := map[string]any{
		"vector":       vector,
		"limit":        limit,
		"with_payload": true,
	}
	raw, err := q.doJSON(ctx, http.MethodPost, q.URL+"/collections/"+collection+"/points/search", body)
	if err != nil {
		return nil, fmt.Errorf("search qdrant: %w", err)
	}

	var res struct {
		Result []struct {
			ID      uint64 `json:"id"`
			Payload struct {
				Text string `json:"text"`
			} `json:"payload"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	points := make([]Point, len(res.Result))
	for i, res := range res.Result {
		points[i] = Point{ID: res.ID, Text: res.Payload.Text}
	}

	return points, nil
}

// doJSON sends a JSON request and returns the raw response body. Non-2xx
// answers become *provider.HTTPError so callers can tell a retryable server
// failure from a permanent rejection.
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
		return nil, &provider.HTTPError{Method: method, URL: url, Status: resp.StatusCode, Body: string(raw)}
	}
	return raw, nil
}
