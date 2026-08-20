// Command ingest wczytuje korpus FlashRAG (jsonl: id, contents) i embeduje go
// wsadowo do kolekcji Qdrant. Ma być uruchamiany raz, osobno od cmd/bench -
// dzięki temu pełny ingest wiki18_100w.jsonl (21M dokumentów) robi się
// jednorazowo, a kolejne biegi cmd/bench odpytują już gotową kolekcję (albo
// wolumen Qdrant przywrócony z kopii - patrz README).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"

	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/rag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

type corpusLine struct {
	ID       string `json:"id"`
	Contents string `json:"contents"`
}

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

func main() {
	var (
		inputPath  = flag.String("input", "dataset/wiki18_100w.jsonl", "ścieżka do pliku korpusu FlashRAG (jsonl: id, contents) do zaingestowania")
		provName   = flag.String("provider", "vllm", "provider embeddera: vllm | ollama")
		embedURL   = flag.String("embed-url", "http://localhost:8001", "URL providera embeddingów")
		embedModel = flag.String("embed-model", "bge-base-en-v1.5", "nazwa modelu embeddingowego")

		storeName   = flag.String("store", "qdrant", "baza wektorowa docelowa: qdrant (na razie jedyna wspierana - flaga przygotowuje pod kolejne backendy)")
		qdrantURL   = flag.String("qdrant-url", "http://localhost:6333", "URL Qdrant")
		collection  = flag.String("collection", "ragbench", "nazwa kolekcji w Qdrant do utworzenia/zapełnienia")
		batchSize   = flag.Int("batch-size", 128, "liczba dokumentów embedowanych i wysyłanych do bazy wektorowej w jednym wsadzie")
		concurrency = flag.Int("concurrency", 8, "liczba wsadów embedowanych i wysyłanych do bazy wektorowej równolegle")
	)
	flag.Parse()

	var embedder Embedder
	switch *provName {
	case "vllm":
		embedder = provider.NewVLLM(*embedURL, *embedModel)
	case "ollama":
		embedder = provider.NewOllama(*embedURL, *embedModel)
	default:
		log.Fatalf("nieznany provider: %s (oczekiwano vllm | ollama)", *provName)
	}

	var store storage.VectorStore
	switch *storeName {
	case "qdrant":
		store = storage.NewQdrant(*qdrantURL)
	default:
		log.Fatalf("nieznany store: %s (obecnie wspierany tylko: qdrant)", *storeName)
	}

	ctx := context.Background()
	pipeline := rag.NewNaiveRAG(store, *collection, embedder, nil)

	f, err := os.Open(*inputPath)
	if err != nil {
		log.Fatalf("otwieranie pliku korpusu: %v", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		log.Fatalf("stat pliku korpusu: %v", err)
	}

	bar := progressbar.DefaultBytes(stat.Size(), "ingest")
	reader := io.TeeReader(f, bar)

	start := time.Now()
	total := 0
	batch := make([]rag.Document, 0, *batchSize)
	collectionReady := false

	ensureCollection := func(dim int) {
		if err := pipeline.EnsureCollection(ctx, dim); err != nil {
			log.Fatalf("tworzenie kolekcji: %v", err)
		}
		collectionReady = true
		log.Printf("kolekcja %q gotowa (wymiar wektora: %d)", *collection, dim)
	}

	sem := make(chan struct{}, *concurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var ingestErr error

	submit := func(docs []rag.Document) {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			if err := pipeline.Ingest(ctx, docs); err != nil {
				mu.Lock()
				if ingestErr == nil {
					ingestErr = err
				}
				mu.Unlock()
			}
		}()
	}

	flush := func() {
		if len(batch) == 0 {
			return
		}
		mu.Lock()
		failed := ingestErr
		mu.Unlock()
		if failed != nil {
			return
		}

		if !collectionReady {
			probe, err := embedder.Embed(ctx, []string{batch[0].Text})
			if err != nil {
				log.Fatalf("wykrywanie wymiaru wektora (probe embed): %v", err)
			}
			ensureCollection(len(probe[0]))
		}

		docs := make([]rag.Document, len(batch))
		copy(docs, batch)
		submit(docs)
		batch = batch[:0]
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var cl corpusLine
		if err := json.Unmarshal([]byte(line), &cl); err != nil {
			log.Fatalf("parsowanie linii korpusu: %v", err)
		}
		id, err := strconv.ParseUint(cl.ID, 10, 64)
		if err != nil {
			log.Fatalf("parsowanie id korpusu %q: %v", cl.ID, err)
		}

		batch = append(batch, rag.Document{ID: id, Text: cl.Contents})
		total++
		if len(batch) >= *batchSize {
			flush()
		}
	}
	flush()
	wg.Wait()
	bar.Finish()

	if err := scanner.Err(); err != nil {
		log.Fatalf("czytanie pliku korpusu: %v", err)
	}
	if ingestErr != nil {
		log.Fatalf("ingest wsadu: %v", ingestErr)
	}

	fmt.Printf("gotowe: %d dokumentów zaingestowanych do kolekcji %q w %s\n", total, *collection, time.Since(start).Round(time.Second))
}
