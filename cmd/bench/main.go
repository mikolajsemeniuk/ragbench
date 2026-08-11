// Command bench uruchamia baseline RAG (NaiveRAG) na zbiorze pytań i mierzy
// jakość generacji (Exact Match, F1) oraz latencję.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/ragbench/pkg/metrics"
	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/rag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

type corpusLine struct {
	ID   uint64 `json:"id"`
	Text string `json:"text"`
}

type datasetLine struct {
	Question string   `json:"question"`
	Answers  []string `json:"answers"`
}

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type Generator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

func main() {
	var (
		providerName = flag.String("provider", "vllm", "provider modeli: vllm | ollama")
		providerURL  = flag.String("provider-url", "http://localhost:8000", "URL providera LLM (generacja)")
		embedURL     = flag.String("embed-url", "http://localhost:8001", "URL providera embeddingów")
		llmModel     = flag.String("llm-model", "qwen2.5-7b-instruct", "nazwa modelu LLM")
		embedModel   = flag.String("embed-model", "bge-base-en-v1.5", "nazwa modelu embeddingowego")

		qdrantURL  = flag.String("qdrant-url", "http://localhost:6333", "URL Qdrant")
		collection = flag.String("collection", "ragbench", "nazwa kolekcji w Qdrant")
		vectorSize = flag.Int("vector-size", 768, "wymiar wektorów embeddingowych")
		topK       = flag.Int("top-k", 5, "liczba fragmentów kontekstu pobieranych przed generacją")

		corpusPath  = flag.String("corpus", "", "ścieżka do pliku korpusu (jsonl: id, text) - opcjonalne, do ingestu przed benchmarkiem")
		datasetPath = flag.String("dataset", "", "ścieżka do pliku pytań (jsonl: question, answers) - wymagane")
		texOut      = flag.String("tex-out", "", "ścieżka pliku .tex do wygenerowania z wynikami (np. paper/baseline.gen.tex) - opcjonalne")
		name        = flag.String("name", "NaiveRAG", "nazwa baseline'u używana w wygenerowanym .tex")
	)
	flag.Parse()

	if *datasetPath == "" {
		log.Fatal("brak wymaganej flagi -dataset")
	}

	ctx := context.Background()

	var embedder Embedder
	var generator Generator
	switch *providerName {
	case "vllm":
		embedder = provider.NewVLLM(*embedURL, *embedModel)
		generator = provider.NewVLLM(*providerURL, *llmModel)
	case "ollama":
		embedder = provider.NewOllama(*embedURL, *embedModel)
		generator = provider.NewOllama(*providerURL, *llmModel)
	default:
		log.Fatalf("nieznany provider: %s (oczekiwano vllm | ollama)", *providerName)
	}

	store := storage.NewQdrant(*qdrantURL)
	pipeline := rag.NewNaiveRAG(store, *collection, embedder, generator)
	pipeline.TopK = *topK

	if *corpusPath != "" {
		docs, err := loadCorpus(*corpusPath)
		if err != nil {
			log.Fatalf("wczytywanie korpusu: %v", err)
		}
		if err := pipeline.EnsureCollection(ctx, *vectorSize); err != nil {
			log.Fatalf("tworzenie kolekcji: %v", err)
		}
		if err := pipeline.Ingest(ctx, docs); err != nil {
			log.Fatalf("ingest korpusu: %v", err)
		}
		log.Printf("zaingestowano %d dokumentów do kolekcji %q", len(docs), *collection)
	}

	dataset, err := loadDataset(*datasetPath)
	if err != nil {
		log.Fatalf("wczytywanie zbioru pytań: %v", err)
	}

	var sumEM, sumF1 float64
	var sumLatency time.Duration
	for i, item := range dataset {
		start := time.Now()
		answer, err := pipeline.Query(ctx, item.Question)
		latency := time.Since(start)
		if err != nil {
			log.Printf("pytanie %d (%q): błąd zapytania: %v", i, item.Question, err)
			continue
		}

		em := metrics.ExactMatch(answer, item.Answers)
		f1 := metrics.F1Score(answer, item.Answers)
		sumEM += em
		sumF1 += f1
		sumLatency += latency

		log.Printf("[%d/%d] EM=%.2f F1=%.2f latency=%s question=%q", i+1, len(dataset), em, f1, latency, item.Question)
	}

	n := float64(len(dataset))
	em := sumEM / n
	f1 := sumF1 / n
	avgLatency := time.Duration(float64(sumLatency) / n)

	fmt.Println("--- wyniki baseline (NaiveRAG) ---")
	fmt.Printf("pytania:         %d\n", len(dataset))
	fmt.Printf("Exact Match:     %.4f\n", em)
	fmt.Printf("F1:              %.4f\n", f1)
	fmt.Printf("Śr. latencja:    %s\n", avgLatency)

	if *texOut != "" {
		if err := writeTex(*texOut, *name, len(dataset), em, f1, avgLatency); err != nil {
			log.Fatalf("zapis pliku tex: %v", err)
		}
		log.Printf("zapisano wyniki do %s", *texOut)
	}
}

// writeTex zapisuje wyniki jako fragment LaTeX (definicje \newcommand), gotowy
// do \input{} w tekście artykułu, np.:
//
//	\input{baseline.gen.tex}
//	Exact Match: \NaiveRAGEM, F1: \NaiveRAGFOne
func writeTex(path, name string, n int, em, f1 float64, avgLatency time.Duration) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}

	id := texSafeID(name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% wygenerowane automatycznie przez cmd/bench - nie edytować ręcznie\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sQuestions}{%d}\n", id, n)
	fmt.Fprintf(&b, "\\newcommand{\\%sEM}{%.4f}\n", id, em)
	fmt.Fprintf(&b, "\\newcommand{\\%sFOne}{%.4f}\n", id, f1)
	fmt.Fprintf(&b, "\\newcommand{\\%sLatency}{%s}\n", id, avgLatency.Round(time.Millisecond))

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// texSafeID usuwa znaki spoza [A-Za-z], bo LaTeX \newcommand dopuszcza w nazwie
// wyłącznie litery.
func texSafeID(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func loadCorpus(path string) ([]rag.Document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var docs []rag.Document
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var cl corpusLine
		if err := json.Unmarshal([]byte(line), &cl); err != nil {
			return nil, fmt.Errorf("parse corpus line: %w", err)
		}
		docs = append(docs, rag.Document{ID: cl.ID, Text: cl.Text})
	}
	return docs, scanner.Err()
}

func loadDataset(path string) ([]datasetLine, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var items []datasetLine
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var dl datasetLine
		if err := json.Unmarshal([]byte(line), &dl); err != nil {
			return nil, fmt.Errorf("parse dataset line: %w", err)
		}
		items = append(items, dl)
	}
	return items, scanner.Err()
}

