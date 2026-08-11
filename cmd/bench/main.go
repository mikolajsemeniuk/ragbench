// Command bench uruchamia baseline RAG (NaiveRAG) na zbiorze pytań i mierzy
// jakość generacji (Exact Match, F1), jakość retrievalu (Recall@K, MRR -
// tylko dla datasetów z adnotacją złotych dokumentów, np. HotpotQA,
// 2WikiMultihopQA, MuSiQue) oraz latencję.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/ragbench/pkg/metrics"
	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/rag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// corpusLine odzwierciedla format korpusu FlashRAG (np. wiki18_100w.jsonl):
// {"id": "0", "contents": "\"Tytuł\"\nTreść fragmentu..."}
type corpusLine struct {
	ID       string `json:"id"`
	Contents string `json:"contents"`
}

// datasetLine odzwierciedla format zbioru pytań FlashRAG. Metadata jest
// zachowywana surowo, bo jej kształt różni się między datasetami - patrz
// extractGoldTitles.
type datasetLine struct {
	Question      string          `json:"question"`
	GoldenAnswers []string        `json:"golden_answers"`
	Metadata      json.RawMessage `json:"metadata"`
}

// datasetMetadata obejmuje dwa kształty adnotacji złotych dokumentów
// spotykane w datasetach FlashRAG:
//   - HotpotQA / 2WikiMultihopQA: metadata.supporting_facts.title
//   - MuSiQue: metadata.question_decomposition[].support_paragraph.title
//
// NaturalQuestions i TriviaQA nie mają takiej adnotacji (open-domain QA bez
// wskazanych złotych fragmentów) - dla nich Recall@K/MRR nie da się policzyć.
type datasetMetadata struct {
	SupportingFacts struct {
		Title []string `json:"title"`
	} `json:"supporting_facts"`
	QuestionDecomposition []struct {
		SupportParagraph struct {
			Title string `json:"title"`
		} `json:"support_paragraph"`
	} `json:"question_decomposition"`
}

// extractGoldTitles zwraca unikalne tytuły złotych dokumentów dla pytania,
// jeśli dataset je adnotuje, w przeciwnym razie nil.
func extractGoldTitles(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}

	var m datasetMetadata
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}

	seen := make(map[string]struct{})
	var titles []string
	add := func(t string) {
		if t == "" {
			return
		}
		if _, ok := seen[t]; ok {
			return
		}
		seen[t] = struct{}{}
		titles = append(titles, t)
	}

	for _, t := range m.SupportingFacts.Title {
		add(t)
	}
	for _, qd := range m.QuestionDecomposition {
		add(qd.SupportParagraph.Title)
	}
	return titles
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
		topK       = flag.Int("top-k", 5, "liczba fragmentów kontekstu pobieranych przed generacją (K dla Recall@K)")

		corpusPath   = flag.String("corpus", "", "ścieżka do pliku korpusu FlashRAG (jsonl: id, contents) - opcjonalne, do ingestu przed benchmarkiem; wymagane też do policzenia Recall@K/MRR (potrzebny indeks tytuł->ID)")
		corpusLimit  = flag.Int("corpus-limit", 0, "0 = zaingestuj cały korpus; N>0 = wylosuj (reservoir sampling, deterministycznie przez -seed) tylko N dokumentów - do szybkiego smoke testu mechaniki pipeline'u. UWAGA: przy małym N złote dokumenty pytań z dużym prawdopodobieństwem nie znajdą się w korpusie, więc EM/F1/Recall@K/MRR będą zaniżone/niereprezentatywne dla jakości - to test 'czy działa', nie benchmark jakości")
		datasetPath  = flag.String("dataset", "", "ścieżka do pliku pytań FlashRAG (jsonl: question, golden_answers[, metadata]) - wymagane. Ewaluację robimy zawsze na splicie dev lub test (np. *_dev.jsonl, *_test.jsonl), nigdy na *_train.jsonl - train w tych datasetach służy wyłącznie jako pula przykładów few-shot, tak jak w FlashRAG/Self-RAG/IRCoT/Adaptive-RAG i innych pracach z README")
		datasetLimit = flag.Int("limit", 0, "0 = użyj całego datasetu; N>0 = wylosuj (bez powtórzeń, deterministycznie przez -seed) tylko N pytań - do szybkiego smoke testu. Losowanie zamiast brania pierwszych N linii, bo niektóre datasety (np. HotpotQA) mają pytania posortowane wg typu/trudności - wzięcie pierwszych N dałoby nierepreznetatywną, przekłamaną próbkę")
		seed         = flag.Int64("seed", 42, "seed losowania dla -limit/-corpus-limit (ta sama wartość => ten sam, powtarzalny sample)")
		texOut       = flag.String("tex-out", "", "ścieżka pliku .tex do wygenerowania z wynikami (np. paper/baseline.gen.tex) - opcjonalne")
		name         = flag.String("name", "NaiveRAG", "nazwa baseline'u używana w wygenerowanym .tex")
	)
	flag.Parse()

	if *datasetPath == "" {
		log.Fatal("brak wymaganej flagi -dataset")
	}
	if strings.Contains(filepath.Base(*datasetPath), "_train") {
		log.Printf("UWAGA: -dataset wskazuje na split train (%s) - kanonicznie ewaluacja QA/RAG odbywa się na dev/test, train służy tylko do few-shot", *datasetPath)
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

	rng := rand.New(rand.NewSource(*seed))

	// titleIndex mapuje tytuł artykułu na ID wszystkich jego fragmentów w
	// korpusie - potrzebne do zamiany złotych tytułów z datasetu na złote ID
	// dokumentów, względem których liczymy Recall@K/MRR.
	var titleIndex map[string][]uint64
	if *corpusPath != "" {
		docs, index, err := loadCorpus(*corpusPath, *corpusLimit, rng)
		if err != nil {
			log.Fatalf("wczytywanie korpusu: %v", err)
		}
		if *corpusLimit > 0 {
			log.Printf("smoke test: zaingestowano losową próbkę %d dokumentów (seed=%d) zamiast całego korpusu", len(docs), *seed)
		}
		titleIndex = index
		if err := pipeline.EnsureCollection(ctx, *vectorSize); err != nil {
			log.Fatalf("tworzenie kolekcji: %v", err)
		}
		if err := pipeline.Ingest(ctx, docs); err != nil {
			log.Fatalf("ingest korpusu: %v", err)
		}
		log.Printf("zaingestowano %d dokumentów do kolekcji %q", len(docs), *collection)
	} else {
		log.Printf("brak -corpus: Recall@K/MRR nie zostaną policzone (brak indeksu tytuł->ID), liczone będą tylko EM/F1/latencja")
	}

	dataset, err := loadDataset(*datasetPath)
	if err != nil {
		log.Fatalf("wczytywanie zbioru pytań: %v", err)
	}
	if *datasetLimit > 0 && *datasetLimit < len(dataset) {
		total := len(dataset)
		rng.Shuffle(total, func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })
		dataset = dataset[:*datasetLimit]
		log.Printf("smoke test: wylosowano %d/%d pytań (seed=%d)", len(dataset), total, *seed)
	}

	var sumEM, sumF1, sumRecall, sumRR float64
	var sumLatency time.Duration
	var retrievalEvalCount int
	for i, item := range dataset {
		start := time.Now()
		answer, retrievedIDs, err := pipeline.Query(ctx, item.Question)
		latency := time.Since(start)
		if err != nil {
			log.Printf("pytanie %d (%q): błąd zapytania: %v", i, item.Question, err)
			continue
		}

		em := metrics.ExactMatch(answer, item.GoldenAnswers)
		f1 := metrics.F1Score(answer, item.GoldenAnswers)
		sumEM += em
		sumF1 += f1
		sumLatency += latency

		logLine := fmt.Sprintf("[%d/%d] EM=%.2f F1=%.2f latency=%s question=%q", i+1, len(dataset), em, f1, latency, item.Question)

		if relevant := goldRelevantIDs(titleIndex, item.Metadata); len(relevant) > 0 {
			recall := metrics.RecallAtK(retrievedIDs, relevant, pipeline.TopK)
			rr := metrics.ReciprocalRank(retrievedIDs, relevant)
			sumRecall += recall
			sumRR += rr
			retrievalEvalCount++
			logLine += fmt.Sprintf(" Recall@%d=%.2f RR=%.2f", pipeline.TopK, recall, rr)
		}

		log.Println(logLine)
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

	var recall, mrr float64
	if retrievalEvalCount > 0 {
		recall = sumRecall / float64(retrievalEvalCount)
		mrr = sumRR / float64(retrievalEvalCount)
		fmt.Printf("Recall@%d:        %.4f (na %d/%d pytań ze złotymi dokumentami)\n", pipeline.TopK, recall, retrievalEvalCount, len(dataset))
		fmt.Printf("MRR:             %.4f (na %d/%d pytań ze złotymi dokumentami)\n", mrr, retrievalEvalCount, len(dataset))
	}

	if *texOut != "" {
		if err := writeTex(*texOut, *name, len(dataset), em, f1, avgLatency, pipeline.TopK, recall, mrr, retrievalEvalCount); err != nil {
			log.Fatalf("zapis pliku tex: %v", err)
		}
		log.Printf("zapisano wyniki do %s", *texOut)
	}
}

// goldRelevantIDs zamienia złote tytuły dokumentów zapisane w metadanych
// pytania na ID fragmentów korpusu (poprzez titleIndex zbudowany przy
// ingestcie). Zwraca nil, jeśli dataset nie adnotuje złotych dokumentów albo
// korpus nie został zaingestowany w tym uruchomieniu.
func goldRelevantIDs(titleIndex map[string][]uint64, rawMetadata json.RawMessage) []uint64 {
	if len(titleIndex) == 0 {
		return nil
	}

	goldTitles := extractGoldTitles(rawMetadata)
	if len(goldTitles) == 0 {
		return nil
	}

	var relevant []uint64
	for _, t := range goldTitles {
		relevant = append(relevant, titleIndex[t]...)
	}
	return relevant
}

// writeTex zapisuje wyniki jako fragment LaTeX (definicje \newcommand), gotowy
// do \input{} w tekście artykułu, np.:
//
//	\input{baseline.gen.tex}
//	Exact Match: \NaiveRAGEM, F1: \NaiveRAGFOne
func writeTex(path, name string, n int, em, f1 float64, avgLatency time.Duration, topK int, recall, mrr float64, retrievalEvalCount int) error {
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
	if retrievalEvalCount > 0 {
		fmt.Fprintf(&b, "\\newcommand{\\%sRecallAtK}{%.4f}\n", id, recall)
		fmt.Fprintf(&b, "\\newcommand{\\%sMRR}{%.4f}\n", id, mrr)
		fmt.Fprintf(&b, "\\newcommand{\\%sTopK}{%d}\n", id, topK)
		fmt.Fprintf(&b, "\\newcommand{\\%sRetrievalEvalQuestions}{%d}\n", id, retrievalEvalCount)
	}

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

// extractTitle wydziela tytuł artykułu z pola contents FlashRAG, którego
// pierwsza linia to tytuł w cudzysłowie, np. "\"Tytuł\"\nTreść...".
func extractTitle(contents string) string {
	line, _, _ := strings.Cut(contents, "\n")
	return strings.Trim(line, "\"")
}

// loadCorpus wczytuje korpus FlashRAG i buduje indeks tytuł->ID fragmentów,
// potrzebny do wyznaczenia złotych ID dokumentów przy liczeniu Recall@K/MRR
// (jeden artykuł Wikipedii jest podzielony na wiele ~100-słowowych
// fragmentów, każdy z tym samym tytułem).
//
// Jeśli limit > 0, zamiast wczytać cały plik (np. 21M linii / 14GB dla
// wiki18_100w.jsonl) stosowane jest reservoir sampling - każda linia ma
// równe prawdopodobieństwo trafienia do próbki, a w pamięci trzymane jest
// najwyżej `limit` dokumentów niezależnie od rozmiaru pliku.
func loadCorpus(path string, limit int, rng *rand.Rand) ([]rag.Document, map[string][]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var docs []rag.Document
	seen := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var cl corpusLine
		if err := json.Unmarshal([]byte(line), &cl); err != nil {
			return nil, nil, fmt.Errorf("parse corpus line: %w", err)
		}
		id, err := strconv.ParseUint(cl.ID, 10, 64)
		if err != nil {
			return nil, nil, fmt.Errorf("parse corpus id %q: %w", cl.ID, err)
		}
		doc := rag.Document{ID: id, Text: cl.Contents}

		if limit <= 0 {
			docs = append(docs, doc)
			continue
		}

		// Algorytm R (reservoir sampling).
		if len(docs) < limit {
			docs = append(docs, doc)
		} else if j := rng.Intn(seen + 1); j < limit {
			docs[j] = doc
		}
		seen++
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}

	titleIndex := make(map[string][]uint64, len(docs))
	for _, d := range docs {
		if title := extractTitle(d.Text); title != "" {
			titleIndex[title] = append(titleIndex[title], d.ID)
		}
	}
	return docs, titleIndex, nil
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
