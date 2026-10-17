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

	// HNSWEf is the size of the candidate list Qdrant keeps while walking the
	// HNSW graph. Left at 0 the server picks its own default, which makes the
	// approximate search's own recall an unreported property of the
	// experiment - and a load-bearing one: a rerank run asks for the top 100
	// and cmd/diagnose for the top 1000, both far beyond the depth a default
	// ef is tuned for, so a passage counted as "not reachable by this query"
	// may only have been missed by the graph walk. Set it explicitly and
	// report it.
	HNSWEf int

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
//
// Setting the value is not enough on its own. Qdrant acknowledges the change
// but can leave the collection in status "grey" - optimizations pending, with
// no optimizer actually running - in which case the HNSW index is never built
// and every search silently falls back to a full scan. Observed on a 21M
// point collection: the threshold read back correctly, indexed_vectors_count
// stayed at 0 for hours at 0.4% CPU, and a single search took 12.7s. Issuing
// the same update again moves the collection to "yellow" and starts the
// build, so the status is verified here and the update repeated until the
// optimizer picks it up.
func (q *Qdrant) SetIndexingThreshold(ctx context.Context, collection string, threshold int) error {
	body := map[string]any{
		"optimizers_config": map[string]any{"indexing_threshold": threshold},
	}

	for attempt := range 3 {
		if _, err := q.doJSON(ctx, http.MethodPatch, q.URL+"/collections/"+collection, body); err != nil {
			return err
		}
		if attempt == 2 || threshold == 0 {
			// A threshold of 0 disables indexing on purpose, so "grey" is the
			// expected outcome and there is nothing to wake up.
			return nil
		}

		status, err := q.collectionStatus(ctx, collection)
		if err != nil || status != "grey" {
			return err
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return nil
}

// collectionStatus returns Qdrant's collection status: green (indexed),
// yellow (optimizing), grey (optimizations pending) or red (error).
func (q *Qdrant) collectionStatus(ctx context.Context, collection string) (string, error) {
	info, err := q.collectionInfo(ctx, collection)
	if err != nil || info == nil {
		return "", err
	}
	return info.Result.Status, nil
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
		Status string `json:"status"`
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
	if q.HNSWEf > 0 {
		body["params"] = map[string]any{"hnsw_ef": q.HNSWEf}
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

// SparseVectorName is the name of the named sparse vector in a lexical
// collection. Qdrant requires sparse vectors to be named even when a
// collection holds only one.
const SparseVectorName = "text"

// EnsureSparseCollection creates a lexical (BM25) collection if it does not
// exist yet.
//
// The collection holds no dense vectors at all ("vectors": {}). That is the
// point: the dense corpus is already indexed in its own collection and cost
// ~3 h of GPU time to build, so lexical search is added next to it rather than
// by rebuilding it. The two are queried separately and fused by the caller.
//
// The "idf" modifier makes Qdrant compute the inverse document frequency of
// every term over the collection at query time. Without it the client would
// have to make a full pass over the corpus first just to count in how many
// passages each term occurs, and would have to redo it whenever the corpus
// changes.
func (q *Qdrant) EnsureSparseCollection(ctx context.Context, collection string) error {
	info, err := q.collectionInfo(ctx, collection)
	if err != nil {
		return fmt.Errorf("check collection exists: %w", err)
	}
	if info != nil {
		return nil
	}

	body := map[string]any{
		"vectors": map[string]any{},
		"sparse_vectors": map[string]any{
			SparseVectorName: map[string]any{
				"index":    map[string]any{"on_disk": true},
				"modifier": "idf",
			},
		},
		"on_disk_payload": true,
	}

	_, err = q.doJSON(ctx, http.MethodPut, q.URL+"/collections/"+collection, body)
	return err
}

func (q *Qdrant) UpsertSparse(ctx context.Context, collection string, points []Point) error {
	if len(points) == 0 {
		return nil
	}

	payload := make([]map[string]any, 0, len(points))
	for _, p := range points {
		// A passage whose every token was filtered out (punctuation only, or
		// nothing but stop words) has no lexical representation. Sending an
		// empty sparse vector is rejected by Qdrant, and storing the point
		// without one would make it unretrievable anyway, so it is skipped -
		// counted by the caller as a skipped document.
		if p.Sparse.Empty() {
			continue
		}
		payload = append(payload, map[string]any{
			"id": p.ID,
			"vector": map[string]any{
				SparseVectorName: map[string]any{
					"indices": p.Sparse.Indices,
					"values":  p.Sparse.Values,
				},
			},
			"payload": map[string]any{"text": p.Text},
		})
	}
	if len(payload) == 0 {
		return nil
	}

	url := q.URL + "/collections/" + collection + "/points"
	if q.Wait {
		url += "?wait=true"
	}
	if _, err := q.doJSON(ctx, http.MethodPut, url, map[string]any{"points": payload}); err != nil {
		return fmt.Errorf("upsert sparse points: %w", err)
	}
	return nil
}

// SearchSparse runs a lexical query through the Query API, which is the
// endpoint that understands sparse vectors (the legacy /points/search does
// not).
func (q *Qdrant) SearchSparse(ctx context.Context, collection string, vector SparseVector, limit int) ([]Point, error) {
	if vector.Empty() {
		return nil, nil
	}

	body := map[string]any{
		"query": map[string]any{
			"nearest": map[string]any{
				"indices": vector.Indices,
				"values":  vector.Values,
			},
		},
		"using":        SparseVectorName,
		"limit":        limit,
		"with_payload": true,
	}
	raw, err := q.doJSON(ctx, http.MethodPost, q.URL+"/collections/"+collection+"/points/query", body)
	if err != nil {
		return nil, fmt.Errorf("sparse search qdrant: %w", err)
	}

	var res struct {
		Result struct {
			Points []struct {
				ID      uint64 `json:"id"`
				Payload struct {
					Text string `json:"text"`
				} `json:"payload"`
			} `json:"points"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("decode sparse search response: %w", err)
	}

	points := make([]Point, len(res.Result.Points))
	for i, r := range res.Result.Points {
		points[i] = Point{ID: r.ID, Text: r.Payload.Text}
	}
	return points, nil
}

// Retrieve fetches points by id. Unlike Search it imposes no ordering: the
// caller already knows which passages it wants and only needs their text.
func (q *Qdrant) Retrieve(ctx context.Context, collection string, ids []uint64) ([]Point, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	body := map[string]any{"ids": ids, "with_payload": true}
	raw, err := q.doJSON(ctx, http.MethodPost, q.URL+"/collections/"+collection+"/points", body)
	if err != nil {
		return nil, fmt.Errorf("retrieve points: %w", err)
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
		return nil, fmt.Errorf("decode retrieve response: %w", err)
	}

	points := make([]Point, len(res.Result))
	for i, r := range res.Result {
		points[i] = Point{ID: r.ID, Text: r.Payload.Text}
	}
	return points, nil
}
