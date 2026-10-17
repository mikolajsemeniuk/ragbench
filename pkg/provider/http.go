// Package provider contains thin HTTP clients for the model servers used by
// the benchmark: vLLM (OpenAI-compatible endpoints) and Ollama.
package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPError is returned when a model server answers with a non-2xx status. It
// keeps the status code so callers can distinguish a permanent, client-side
// rejection (4xx - the request itself is unacceptable and resending it
// unchanged will fail again) from a transient, server-side failure (5xx, 429
// - worth retrying). The full response body is kept because model servers put
// the actionable detail there, e.g. the exact token count that overflowed the
// context window.
type HTTPError struct {
	Method string
	URL    string
	Status int
	Body   string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("%s %s returned status %d: %s", e.Method, e.URL, e.Status, e.Body)
}

// Retryable reports whether resending the identical request may succeed.
func (e *HTTPError) Retryable() bool {
	switch e.Status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests:
		return true
	}
	return e.Status >= 500
}

// IsRetryable classifies an arbitrary error from this package. Transport-level
// failures (connection refused while the server restarts, read timeout, reset
// connection) are retryable; a context cancellation - the operator pressing
// Ctrl-C - is not.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		return httpErr.Retryable()
	}
	// Anything that is not a decoded HTTP response is a transport or decoding
	// failure, both of which are transient in practice.
	return true
}

// NewHTTPClient builds a client suited to a high-concurrency ingest run.
//
// http.DefaultClient is deliberately not used: it has no timeout at all (a
// stalled server would hang a worker forever) and http.DefaultTransport caps
// idle connections at 2 per host, so every request beyond the first two tears
// down and re-establishes a TCP connection - which, at millions of requests,
// dominates the runtime and exhausts local ports.
func NewHTTPClient(timeout time.Duration, maxIdleConnsPerHost int) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = maxIdleConnsPerHost * 2
	transport.MaxIdleConnsPerHost = maxIdleConnsPerHost
	transport.IdleConnTimeout = 90 * time.Second

	return &http.Client{Transport: transport, Timeout: timeout}
}

// doJSON sends a JSON request and returns the raw response body.
func doJSON(ctx context.Context, client *http.Client, method, url string, in any) ([]byte, error) {
	body, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	if client == nil {
		client = http.DefaultClient
	}
	res, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer res.Body.Close()

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if res.StatusCode >= 300 {
		return nil, &HTTPError{Method: method, URL: url, Status: res.StatusCode, Body: truncateForError(string(raw))}
	}
	return raw, nil
}

// truncateForError keeps error messages readable when a server echoes back a
// large request body.
func truncateForError(s string) string {
	const max = 512
	if len(s) <= max {
		return s
	}
	return s[:max] + "... (truncated)"
}
