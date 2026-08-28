// Command ingest loads a FlashRAG corpus (jsonl: id, contents), embeds it in
// batches and upserts it into a Qdrant collection. It is meant to be run once,
// separately from cmd/bench: the full wiki18_100w.jsonl ingest (~21M passages)
// happens a single time, and later cmd/bench runs query the ready collection
// (or a Qdrant volume restored from a backup - see README).
//
// The run is designed to be restartable and to never report success it did not
// achieve:
//
//   - Points are keyed by the corpus id, so upserts are idempotent: re-running
//     the same range overwrites rather than duplicates, and an interrupted run
//     can be resumed with -skip without cleaning up first.
//   - Every request is retried with exponential backoff, which covers an
//     embedding server that crashes and is restarted by Docker mid-run.
//   - A batch rejected outright (4xx) is bisected down to the offending
//     document instead of being dropped whole.
//   - If the embedding server stays down, the run aborts loudly instead of
//     spinning for hours while nothing reaches the database.
//   - The final document count is verified against Qdrant, and the process
//     exits non-zero if anything was lost.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/schollz/progressbar/v3"

	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/rag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// corpusLine mirrors the FlashRAG corpus format:
// {"id": "0", "contents": "\"Title\"\nPassage text..."}
type corpusLine struct {
	ID       string `json:"id"`
	Contents string `json:"contents"`
}

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

func main() {
	var (
		inputPath  = flag.String("input", "dataset/wiki18_100w.jsonl", "path to the FlashRAG corpus file (jsonl: id, contents) to ingest")
		provName   = flag.String("provider", "vllm", "embedding provider: vllm | ollama")
		embedURL   = flag.String("embed-url", "http://localhost:8001", "embedding provider URL")
		embedModel = flag.String("embed-model", "bge-base-en-v1.5", "embedding model name")

		sparse = flag.Bool("sparse", false, "build a lexical BM25 index instead of a dense one. It needs no embedding server and no GPU: the vectors are term frequencies computed locally, and Qdrant supplies the inverse document frequency at query time. Use a different -collection from the dense one - the two are separate collections over the same passage ids, fused at query time by cmd/bench -architecture hybrid")

		storeName  = flag.String("store", "qdrant", "target vector database: qdrant (the only one supported so far - the flag reserves room for further backends)")
		qdrantURL  = flag.String("qdrant-url", "http://localhost:6333", "Qdrant URL")
		collection = flag.String("collection", "ragbench", "name of the Qdrant collection to create/fill")

		batchSize   = flag.Int("batch-size", 128, "number of documents embedded and upserted in one batch")
		concurrency = flag.Int("concurrency", 8, "number of batches embedded and upserted in parallel")

		maxTokens = flag.Int("max-tokens", 512, "context limit passed to the embedding model; longer documents are truncated to it. With -provider vllm the model's own tokenizer performs the cut server-side (truncate_prompt_tokens), so the limit is exact; keep it at or below the model's max_model_len")
		maxWords  = flag.Int("max-words", 400, "fallback word limit, used only when the provider cannot truncate server-side (-provider ollama)")

		docPrefix = flag.String("doc-prefix", rag.PrefixAuto, "instruction prepended to each passage before embedding it. \"auto\" uses the convention documented for -embed-model (BGE: none; E5: \"passage: \"; Nomic Embed: \"search_document: \"). It is baked into the stored vectors, so changing it invalidates the collection and requires a full re-ingest; cmd/bench must be run with the matching -query-prefix. The prefix is an instruction to the encoder and is not stored in the payload")

		skip  = flag.Int64("skip", 0, "skip the first N corpus lines - used to resume an interrupted run (upserts are keyed by corpus id and therefore idempotent)")
		limit = flag.Int64("limit", 0, "0 = ingest to the end of the file; N>0 = stop after N documents (smoke test)")
		total = flag.Int64("total", 0, "0 = count the corpus lines up front for an accurate progress bar and final verification; N>0 = trust N instead (skips a full pass over the file)")

		httpTimeout   = flag.Duration("http-timeout", 5*time.Minute, "per-request timeout for the embedding server and Qdrant")
		maxRetries    = flag.Int("max-retries", 8, "retries per request on transient failures (5xx, 429, connection errors); with the default backoff this tolerates an embedding server being down for roughly 8 minutes, i.e. a container restart")
		retryBase     = flag.Duration("retry-base", 2*time.Second, "initial backoff delay; doubles with each retry, with jitter")
		retryMax      = flag.Duration("retry-max", 2*time.Minute, "backoff delay cap")
		maxConsecFail = flag.Int64("max-consecutive-failures", 10, "abort the run after this many batches fail with their retries exhausted - stops a run from continuing for hours against a dead embedding server while nothing is stored")

		indexingThreshold = flag.Int("indexing-threshold", 20000, "Qdrant indexing_threshold restored after the ingest, which triggers HNSW index construction. During the ingest it is held at 0 so index building does not compete with the load")
		finalize          = flag.Bool("finalize", true, "after a complete ingest, restore -indexing-threshold and verify the point count against the corpus")
		verify            = flag.Bool("verify", true, "compare the resulting Qdrant point count with the number of ingested documents")
	)
	flag.Parse()

	if *batchSize < 1 {
		log.Fatalf("-batch-size must be >= 1, got %d", *batchSize)
	}
	if *concurrency < 1 {
		log.Fatalf("-concurrency must be >= 1, got %d", *concurrency)
	}
	if *maxTokens < 1 {
		log.Fatalf("-max-tokens must be >= 1, got %d", *maxTokens)
	}

	// Ctrl-C (and SIGTERM) cancel the context, which unwinds the in-flight
	// requests and prints the -skip value to resume from, instead of killing
	// the process mid-batch.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var embedder Embedder
	// truncateLocally is the pre-truncation applied before a document is sent.
	// It is a no-op for providers that truncate exactly on their own.
	truncateLocally := func(text string) string { return text }

	switch *provName {
	case "vllm":
		v := provider.NewVLLM(*embedURL, *embedModel)
		v.Client = provider.NewHTTPClient(*httpTimeout, *concurrency*2)
		// Let the model's tokenizer do the truncation on the exact token
		// sequence it is about to embed. Without this, a single over-long
		// passage makes vLLM reject the whole batch with 400 ("This model's
		// maximum context length is 512 tokens"), and no client-side word or
		// token estimate is tight enough to prevent it reliably.
		v.TruncatePromptTokens = *maxTokens
		embedder = v
	case "ollama":
		o := provider.NewOllama(*embedURL, *embedModel)
		o.Client = provider.NewHTTPClient(*httpTimeout, *concurrency*2)
		embedder = o
		// Ollama offers no server-side truncation, so the word count is the
		// only bound available. It is a conservative approximation: English
		// text averages more than one token per word, so a word limit below
		// the token limit is safe.
		truncateLocally = func(text string) string {
			t, _ := truncateWords(text, *maxWords)
			return t
		}
	default:
		log.Fatalf("unknown provider: %s (expected vllm | ollama)", *provName)
	}

	var store storage.VectorStore
	var qdrant *storage.Qdrant
	switch *storeName {
	case "qdrant":
		qdrant = storage.NewQdrant(*qdrantURL)
		qdrant.Client = provider.NewHTTPClient(*httpTimeout, *concurrency*2)
		store = qdrant
	default:
		log.Fatalf("unknown store: %s (currently supported: qdrant)", *storeName)
	}

	// The two modes differ only in what a batch turns into: a dense vector
	// from the embedding server, or a bag of term frequencies computed here.
	// Everything downstream - batching, retries, bisection, resume,
	// verification - is shared.
	var (
		pipeline      batchIngester
		bm25          *rag.BM25Ingester
		docPrefixInfo string
	)
	if *sparse {
		bm25 = rag.NewBM25Ingester(qdrant, *collection)
		pipeline = bm25
		docPrefixInfo = "n/a (lexical index, no encoder)"
		truncateLocally = func(text string) string { return text }
	} else {
		dense := rag.NewNaiveRAG(store, *collection, embedder, nil)
		dense.DocumentPrefix = rag.ResolvePrefix(*docPrefix, *embedModel, false)
		if _, _, known := rag.PrefixesFor(*embedModel); !known && *docPrefix == rag.PrefixAuto {
			log.Printf("document prefix: none (no convention known for embedding model %q; pass -doc-prefix explicitly if it expects one)", *embedModel)
		} else {
			log.Printf("document prefix: %q", dense.DocumentPrefix)
		}
		pipeline = dense
		docPrefixInfo = fmt.Sprintf("%q", dense.DocumentPrefix)
	}

	in := &ingester{
		pipeline:      pipeline,
		maxRetries:    *maxRetries,
		retryBase:     *retryBase,
		retryMax:      *retryMax,
		maxConsecFail: *maxConsecFail,
		inFlight:      make(map[int64]struct{}),
	}

	var dim int
	if *sparse {
		if err := qdrant.EnsureSparseCollection(ctx, *collection); err != nil {
			log.Fatalf("creating sparse collection: %v", err)
		}
		log.Printf("collection %q ready (sparse BM25 index on disk, inverse document frequency computed by Qdrant)", *collection)
	} else {
		// Probe the embedding server before touching the corpus: this both
		// fails fast with a readable message when the server is not up, and
		// yields the vector dimension needed to create the collection. The
		// dimension of an embedding model does not depend on the input, so a
		// short fixed string is enough.
		err := in.withRetry(ctx, "probe embedding server", func() error {
			vectors, err := embedder.Embed(ctx, []string{"dimension probe"})
			if err != nil {
				return err
			}
			dim = len(vectors[0])
			return nil
		})
		if err != nil {
			log.Fatalf("embedding server %s (model %q) is not usable: %v", *embedURL, *embedModel, err)
		}
		log.Printf("embedding model %q at %s returns %d-dimensional vectors", *embedModel, *embedURL, dim)

		cfg := storage.DefaultCollectionConfig(dim)
		bulkThreshold := 0
		cfg.IndexingThreshold = &bulkThreshold
		if err := qdrant.EnsureCollectionWithConfig(ctx, *collection, cfg); err != nil {
			log.Fatalf("creating collection: %v", err)
		}
		log.Printf("collection %q ready (vector size: %d, distance: %s, vectors and payload on disk, indexing_threshold=0 for the load)", *collection, dim, cfg.Distance)
	}

	f, err := os.Open(*inputPath)
	if err != nil {
		log.Fatalf("opening corpus file: %v", err)
	}
	defer f.Close()

	expected := *total
	if expected <= 0 {
		log.Printf("counting corpus lines in %s (pass -total to skip this)", *inputPath)
		expected, err = countLines(*inputPath)
		if err != nil {
			log.Fatalf("counting corpus lines: %v", err)
		}
	}
	remaining := expected - *skip
	if remaining < 0 {
		remaining = 0
	}
	if *limit > 0 && *limit < remaining {
		remaining = *limit
	}
	log.Printf("corpus has %d documents; this run will process %d of them (skip=%d, limit=%d)", expected, remaining, *skip, *limit)

	bar := progressbar.NewOptions64(remaining,
		progressbar.OptionSetDescription("ingest"),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("docs"),
		progressbar.OptionThrottle(200*time.Millisecond),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)
	in.bar = bar

	start := time.Now()

	// A small buffer in front of the workers keeps them fed while the reader
	// parses the next batch, without letting the reader run arbitrarily far
	// ahead of what has actually been stored.
	jobs := make(chan job, *concurrency)
	var wg sync.WaitGroup
	for range *concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				// A cancelled run still drains the channel so the reader never
				// blocks, but leaves the batch marked in flight: resumeFrom must
				// report a -skip value that covers it.
				if ctx.Err() != nil {
					continue
				}
				if err := in.ingestBatch(ctx, j.docs); err != nil {
					if ctx.Err() == nil {
						in.recordHardFailure(err)
					}
					continue
				}
				in.doneWith(j.firstLine)
			}
		}()
	}

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)

	var (
		lineNo          int64 // 1-based number of the line just read
		processed       int64
		batch           = make([]rag.Document, 0, *batchSize)
		batchFirst      int64
		readErr         error
		interruptedRead bool
	)

	dispatch := func() {
		if len(batch) == 0 {
			return
		}
		docs := make([]rag.Document, len(batch))
		copy(docs, batch)
		in.startingWith(batchFirst)
		jobs <- job{docs: docs, firstLine: batchFirst}
		batch = batch[:0]
	}

readLoop:
	for scanner.Scan() {
		lineNo++
		if lineNo <= *skip {
			continue
		}
		select {
		case <-ctx.Done():
			interruptedRead = true
			break readLoop
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var cl corpusLine
		if err := json.Unmarshal([]byte(line), &cl); err != nil {
			readErr = fmt.Errorf("parsing corpus line %d: %w", lineNo, err)
			break readLoop
		}
		id, err := strconv.ParseUint(cl.ID, 10, 64)
		if err != nil {
			readErr = fmt.Errorf("parsing corpus id %q on line %d: %w", cl.ID, lineNo, err)
			break readLoop
		}

		if len(batch) == 0 {
			batchFirst = lineNo
		}
		// Note: the text stored in the payload is the full passage. Only the
		// embedding is computed over the first -max-tokens tokens. That is the
		// usual arrangement (the index is truncated, the context handed to the
		// generator is not) and it matters here because the previous version
		// wrote the truncated text back into Qdrant, silently corrupting the
		// corpus it was supposed to store.
		batch = append(batch, rag.Document{ID: id, Text: truncateLocally(cl.Contents)})
		processed++
		bar.Add(1)

		if len(batch) >= *batchSize {
			dispatch()
		}
		if *limit > 0 && processed >= *limit {
			break readLoop
		}
	}
	if readErr == nil {
		if err := scanner.Err(); err != nil {
			readErr = fmt.Errorf("reading corpus file: %w", err)
		}
	}
	dispatch()
	close(jobs)
	wg.Wait()
	_ = bar.Finish()
	fmt.Fprintln(os.Stderr)

	elapsed := time.Since(start)
	interrupted := ctx.Err() != nil || interruptedRead
	aborted := in.aborted.Load()

	if readErr != nil {
		log.Printf("corpus read stopped: %v", readErr)
	}

	fmt.Printf("\n--- ingest summary ---\n")
	fmt.Printf("corpus:               %s\n", *inputPath)
	fmt.Printf("collection:           %s (%s)\n", *collection, *qdrantURL)
	if *sparse {
		fmt.Printf("index:                lexical BM25 (k1=%.1f, b=%.2f, avgdl=%.0f), no embedding model used\n", rag.NewBM25().K1, rag.NewBM25().B, rag.NewBM25().AvgDocLen)
		fmt.Printf("documents with no indexable term: %d\n", bm25.Empty.Load())
	} else {
		fmt.Printf("embedding model:      %s (%s, %d dims, truncated to %d tokens)\n", *embedModel, *provName, dim, *maxTokens)
	}
	fmt.Printf("document prefix:      %s\n", docPrefixInfo)
	fmt.Printf("documents read:       %d\n", processed)
	fmt.Printf("documents ingested:   %d\n", in.ingested.Load())
	fmt.Printf("documents shrunk:     %d (rejected individually, retried with less text)\n", in.shrunk.Load())
	fmt.Printf("documents skipped:    %d\n", in.skipped.Load())
	fmt.Printf("batches bisected:     %d (a rejected batch split in half rather than dropped)\n", in.bisected.Load())
	fmt.Printf("requests retried:     %d\n", in.retried.Load())
	fmt.Printf("batches failed:       %d (retries exhausted)\n", in.hardFailures.Load())
	fmt.Printf("elapsed:              %s\n", elapsed.Round(time.Second))
	if processed > 0 && elapsed > 0 {
		fmt.Printf("throughput:           %.0f docs/s\n", float64(in.ingested.Load())/elapsed.Seconds())
	}

	complete := readErr == nil && !interrupted && !aborted &&
		in.hardFailures.Load() == 0 && in.skipped.Load() == 0 &&
		in.ingested.Load() == processed

	// Only the dense collection was loaded with indexing disabled; Qdrant
	// builds a sparse index inline, so there is nothing to restore for it.
	if complete && *finalize && !*sparse {
		if err := qdrant.SetIndexingThreshold(context.Background(), *collection, *indexingThreshold); err != nil {
			log.Printf("restoring indexing_threshold=%d failed: %v", *indexingThreshold, err)
		} else {
			fmt.Printf("indexing_threshold:   restored to %d (Qdrant now builds the HNSW index in the background)\n", *indexingThreshold)
		}
	}

	if *verify {
		count, err := qdrant.CountPoints(context.Background(), *collection)
		if err != nil {
			log.Printf("verifying point count failed: %v", err)
		} else {
			fmt.Printf("points in collection: %d\n", count)
			if complete && *skip == 0 && *limit == 0 && count != expected {
				fmt.Printf("WARNING: the collection holds %d points but the corpus has %d lines. Qdrant keys points by corpus id, so a lower number means the corpus contains duplicate ids.\n", count, expected)
			}
		}
	}

	switch {
	case aborted:
		fmt.Fprintf(os.Stderr, "\nABORTED: %d consecutive batches failed with their retries exhausted. The embedding server or Qdrant is down - check `docker compose logs vllm-embed`, then resume with -skip %d.\n", *maxConsecFail, in.resumeFrom(lineNo))
		os.Exit(1)
	case interrupted:
		fmt.Fprintf(os.Stderr, "\nINTERRUPTED. Resume with -skip %d (upserts are keyed by corpus id, so re-running an overlapping range is safe).\n", in.resumeFrom(lineNo))
		os.Exit(1)
	case readErr != nil:
		fmt.Fprintf(os.Stderr, "\nFAILED while reading the corpus: %v\n", readErr)
		os.Exit(1)
	case in.hardFailures.Load() > 0 || in.skipped.Load() > 0 || in.ingested.Load() != processed:
		fmt.Fprintf(os.Stderr, "\nINCOMPLETE: %d of %d documents reached the collection. Re-run to fill the gaps (upserts are idempotent).\n", in.ingested.Load(), processed)
		os.Exit(1)
	default:
		fmt.Printf("\nOK: all %d documents ingested.\n", processed)
	}
}

// job is one batch handed to a worker. firstLine is the corpus line number of
// its first document, used to compute a safe -skip value if the run stops
// early.
type job struct {
	docs      []rag.Document
	firstLine int64
}

// batchIngester stores one batch of documents. It is an interface so that the
// dense and the lexical ingest share every piece of machinery around it.
type batchIngester interface {
	Ingest(ctx context.Context, docs []rag.Document) error
}

type ingester struct {
	pipeline batchIngester
	bar      *progressbar.ProgressBar

	maxRetries    int
	retryBase     time.Duration
	retryMax      time.Duration
	maxConsecFail int64

	ingested     atomic.Int64
	skipped      atomic.Int64
	shrunk       atomic.Int64
	bisected     atomic.Int64
	retried      atomic.Int64
	hardFailures atomic.Int64
	consecFail   atomic.Int64
	aborted      atomic.Bool
	logged       atomic.Int64

	mu       sync.Mutex
	inFlight map[int64]struct{}
}

// maxLoggedProblems bounds the diagnostic output: on a 21M-document corpus a
// systematic failure would otherwise produce millions of identical lines. The
// summary at the end reports the full counts.
const maxLoggedProblems = 30

func (in *ingester) logf(format string, args ...any) {
	if in.logged.Add(1) > maxLoggedProblems {
		return
	}
	if in.bar != nil {
		_ = in.bar.Clear()
	}
	log.Printf(format, args...)
	if in.logged.Load() == maxLoggedProblems {
		log.Printf("(further per-batch diagnostics suppressed; see the summary at the end)")
	}
}

// startingWith / doneWith track which corpus lines are in flight. A batch is
// only cleared once it is fully stored, so an interrupted or aborted run can
// report a -skip value that is guaranteed not to leave a gap.
func (in *ingester) startingWith(firstLine int64) {
	in.mu.Lock()
	in.inFlight[firstLine] = struct{}{}
	in.mu.Unlock()
}

func (in *ingester) doneWith(firstLine int64) {
	in.mu.Lock()
	delete(in.inFlight, firstLine)
	in.mu.Unlock()
}

// resumeFrom returns the number of lines that are safe to skip on a re-run:
// everything before the earliest batch that was still in flight. nextLine is
// the last line the reader consumed.
func (in *ingester) resumeFrom(nextLine int64) int64 {
	in.mu.Lock()
	defer in.mu.Unlock()

	lowest := nextLine
	for line := range in.inFlight {
		if line-1 < lowest {
			lowest = line - 1
		}
	}
	if lowest < 0 {
		return 0
	}
	return lowest
}

// withRetry runs fn, retrying transient failures with exponential backoff and
// jitter. Permanent failures (4xx) return immediately - resending an
// unacceptable request unchanged cannot succeed.
func (in *ingester) withRetry(ctx context.Context, what string, fn func() error) error {
	var lastErr error
	for attempt := 0; ; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !provider.IsRetryable(lastErr) || attempt >= in.maxRetries {
			break
		}

		in.retried.Add(1)
		delay := in.backoff(attempt)
		in.logf("%s failed (attempt %d/%d), retrying in %s: %v", what, attempt+1, in.maxRetries+1, delay.Round(time.Millisecond), lastErr)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}

	if !provider.IsRetryable(lastErr) {
		return lastErr
	}
	return fmt.Errorf("%s: giving up after %d attempts: %w", what, in.maxRetries+1, lastErr)
}

// backoff returns retryBase * 2^attempt, capped at retryMax, with 50-100%
// jitter so that concurrent workers recovering from the same outage do not
// resend in lockstep.
func (in *ingester) backoff(attempt int) time.Duration {
	delay := in.retryBase
	for range attempt {
		delay *= 2
		if delay >= in.retryMax {
			delay = in.retryMax
			break
		}
	}
	return time.Duration(float64(delay) * (0.5 + 0.5*rand.Float64()))
}

// ingestBatch embeds and upserts one batch.
//
// A batch can fail for two very different reasons and they need opposite
// responses. A transient failure (the embedding server restarting, Qdrant
// briefly overloaded) affects every document equally and is handled by
// retrying the batch as a whole. A permanent rejection (4xx) is usually caused
// by one pathological document - historically an over-long passage - and the
// fix is to bisect the batch until the culprit is isolated, so the other 255
// perfectly good documents are still stored. The previous implementation
// dropped the whole batch on any error, which is how entire runs ended with an
// empty collection.
func (in *ingester) ingestBatch(ctx context.Context, docs []rag.Document) error {
	if len(docs) == 0 {
		return nil
	}

	what := fmt.Sprintf("ingest batch of %d (ids %d-%d)", len(docs), docs[0].ID, docs[len(docs)-1].ID)
	err := in.withRetry(ctx, what, func() error { return in.pipeline.Ingest(ctx, docs) })
	if err == nil {
		in.ingested.Add(int64(len(docs)))
		in.consecFail.Store(0)
		return nil
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if provider.IsRetryable(err) {
		// Retries are exhausted, so the server is down rather than the data
		// being bad. Bisecting here would only multiply the load on a service
		// that is already failing.
		return err
	}

	if len(docs) > 1 {
		in.bisected.Add(1)
		in.logf("%s rejected, bisecting: %v", what, err)
		mid := len(docs) / 2
		return errors.Join(
			in.ingestBatch(ctx, docs[:mid]),
			in.ingestBatch(ctx, docs[mid:]),
		)
	}
	return in.ingestSingle(ctx, docs[0], err)
}

// ingestSingle handles a document the server rejects on its own. It halves the
// text repeatedly and retries. With -provider vllm this is dead weight, since
// the server truncates exactly; it is the safety net for providers that
// cannot (Ollama) and for rejections that are about payload size rather than
// token count.
func (in *ingester) ingestSingle(ctx context.Context, doc rag.Document, cause error) error {
	text := doc.Text
	for range 6 {
		words := len(strings.Fields(text))
		if words <= 1 {
			break
		}
		shorter, _ := truncateWords(text, words/2)
		text = shorter

		attempt := rag.Document{ID: doc.ID, Text: text}
		err := in.withRetry(ctx, fmt.Sprintf("ingest document %d shortened to %d words", doc.ID, words/2), func() error {
			return in.pipeline.Ingest(ctx, []rag.Document{attempt})
		})
		if err == nil {
			in.ingested.Add(1)
			in.shrunk.Add(1)
			in.consecFail.Store(0)
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if provider.IsRetryable(err) {
			return err
		}
	}

	in.skipped.Add(1)
	in.logf("document %d skipped, rejected even after shortening: %v", doc.ID, cause)
	return nil
}

// recordHardFailure counts a batch whose retries ran out and trips the circuit
// breaker once too many happen back to back. Without it a dead embedding
// server produces a run that looks busy for hours and stores nothing - the
// exact failure mode this command was rewritten to eliminate.
func (in *ingester) recordHardFailure(err error) {
	in.hardFailures.Add(1)
	in.logf("batch failed permanently: %v", err)
	if in.consecFail.Add(1) >= in.maxConsecFail {
		if in.aborted.CompareAndSwap(false, true) {
			if in.bar != nil {
				_ = in.bar.Clear()
			}
			log.Printf("aborting: %d consecutive batches failed with retries exhausted", in.maxConsecFail)
		}
	}
}

// truncateWords trims text to at most maxWords words. Used only as an
// approximation for providers without server-side truncation: English text
// averages more than one token per word, so a word limit below the model's
// token limit is conservative.
func truncateWords(text string, maxWords int) (string, bool) {
	words := strings.Fields(text)
	if len(words) <= maxWords {
		return text, false
	}
	return strings.Join(words[:maxWords], " "), true
}

// countLines counts the lines in a file in one sequential pass, for the
// progress bar and the final verification against Qdrant.
func countLines(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var count int64
	buf := make([]byte, 4*1024*1024)
	for {
		n, err := f.Read(buf)
		count += int64(bytes.Count(buf[:n], []byte{'\n'}))
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, err
		}
	}
	return count, nil
}
