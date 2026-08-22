// Command ingest wczytuje korpus FlashRAG (jsonl: id, contents) i embeduje go
// wsadowo do kolekcji Qdrant. Ma być uruchamiany raz, osobno od cmd/bench -
// dzięki temu pełny ingest wiki18_100w.jsonl (21M dokumentów) robi się
// jednorazowo, a kolejne biegi cmd/bench odpytują już gotową kolekcję (albo
// wolumen Qdrant przywrócony z kopii - patrz README).
package main

import (
	"bufio"
	"bytes"
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
	"sync/atomic"
	"time"

	"github.com/schollz/progressbar/v3"

	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/rag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// countLines liczy liczbę linii w pliku (potrzebne do paska postępu opartego
// o liczbę dokumentów, a nie o bajty). Robi to jednym szybkim przejściem po
// pliku, licząc bajty '\n'.
func countLines(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var count int64
	buf := make([]byte, 1024*1024)
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

// truncateWords przycina tekst do co najwyżej maxWords słów. Modele
// embeddingowe mają twardy limit długości kontekstu (np. bge-base-en-v1.5:
// 512 tokenów); ponieważ nie liczymy tu dokładnej liczby tokenów (brak
// tokenizera BPE po stronie Go), używamy liczby słów jako bezpiecznego,
// konserwatywnego przybliżenia (średnio 1 token odpowiada mniej niż 1
// słowu w języku angielskim, więc limit słów niższy niż limit tokenów jest
// bezpieczny). Zwraca przycięty tekst oraz informację, czy przycięcie
// nastąpiło.
func truncateWords(text string, maxWords int) (string, bool) {
	words := strings.Fields(text)
	if len(words) <= maxWords {
		return text, false
	}
	return strings.Join(words[:maxWords], " "), true
}

type corpusLine struct {
	ID       string `json:"id"`
	Contents string `json:"contents"`
}

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Tokenizer to opcjonalna zdolność embeddera do dokładnego przycinania
// tekstu do limitu tokenów jego własnego tokenizera (patrz
// provider.VLLM.TruncateToTokens). Providerzy, którzy jej nie implementują
// (np. Ollama), dostają przybliżenie po liczbie słów (truncateWords) jako
// fallback.
type Tokenizer interface {
	TruncateToTokens(ctx context.Context, text string, maxTokens int) (string, bool, error)
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
		maxTokens   = flag.Int("max-tokens", 512, "maksymalna liczba tokenów dokumentu przekazywana do embeddera; dłuższe dokumenty są przycinane do tej długości (dokładnie, przez tokenizer modelu, jeśli provider to wspiera - patrz Tokenizer)")
		maxWords    = flag.Int("max-words", 400, "fallback: maksymalna liczba słów dokumentu, używana tylko gdy provider nie wspiera dokładnej tokenizacji (np. Ollama) - przybliżenie limitu tokenów")
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

	// truncate przycina tekst dokumentu do limitu kontekstu embeddera.
	// Woli dokładną tokenizację providera (tokenizer.TruncateToTokens),
	// a dopiero gdy provider jej nie wspiera - przybliżenie po liczbie słów
	// (maxTokens jest wtedy używane jako przybliżony limit słów).
	truncate := func(text string, maxTokens int) (string, bool, error) {
		if tk, ok := embedder.(Tokenizer); ok {
			return tk.TruncateToTokens(ctx, text, maxTokens)
		}
		t, wasTruncated := truncateWords(text, maxTokens)
		return t, wasTruncated, nil
	}

	// Wołanie dokładnej tokenizacji dla każdego z 21M dokumentów byłoby
	// niepotrzebnie kosztowne (dodatkowy round-trip HTTP na dokument), skoro
	// korpus FlashRAG jest już wstępnie podzielony na fragmenty ~100-słowne
	// i zdecydowana większość dokumentów jest krótka. Dlatego stosujemy
	// tani lokalny pre-filtr po liczbie słów i dokładną, kosztowną
	// tokenizację przez sieć wywołujemy tylko dla dokumentów, które go
	// przekraczają. Próg musi mieć spory margines poniżej *maxWords: sam
	// *maxWords bywa niewystarczającym przybliżeniem (np. 400 słów dawało w
	// praktyce 513 tokenów przez interpunkcję/liczby rozbijane przez BPE na
	// więcej niż jeden token), więc próg pre-filtra jest wyraźnie niższy, by
	// żaden dokument bliski granicy nie ominął dokładnej tokenizacji.
	wordPrefilterThreshold := *maxWords / 2

	f, err := os.Open(*inputPath)
	if err != nil {
		log.Fatalf("otwieranie pliku korpusu: %v", err)
	}
	defer f.Close()

	lineCount, err := countLines(*inputPath)
	if err != nil {
		log.Fatalf("liczenie linii korpusu: %v", err)
	}

	bar := progressbar.NewOptions64(lineCount,
		progressbar.OptionSetDescription("ingest"),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("docs"),
	)

	start := time.Now()
	total := 0
	var truncated atomic.Int64
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
	var failedBatches atomic.Int64
	var skippedDocs atomic.Int64

	// ingestOne wsadza pojedynczy dokument, dociążając go coraz mocniej, gdy
	// embedder wciąż go odrzuca (np. przejściowy błąd sieci, albo dokument
	// wciąż zbyt długi mimo wstępnego przycięcia - patrz truncate powyżej).
	// Używane jako fallback po nieudanym wsadzie zbiorczym.
	ingestOne := func(doc rag.Document) error {
		tokens := *maxTokens
		var lastErr error
		for attempt := 0; attempt < 4; attempt++ {
			if err := pipeline.Ingest(ctx, []rag.Document{doc}); err != nil {
				lastErr = err
				tokens /= 2
				if tokens < 1 {
					break
				}
				if t, _, terr := truncate(doc.Text, tokens); terr == nil {
					doc.Text = t
				} else {
					doc.Text, _ = truncateWords(doc.Text, tokens)
				}
				continue
			}
			return nil
		}
		return lastErr
	}

	// submit wysyła jeden wsad asynchronicznie. Gdy zbiorczy request padnie
	// (np. jeden dokument w wsadzie wciąż przekracza limit tokenów mimo
	// wstępnego przycięcia po słowach), reszta wsadu NIE jest tracona -
	// program wsadza dokumenty pojedynczo i dociąga tylko winowajcę/winowajców,
	// zamiast porzucać cały wsad poprawnych dokumentów.
	submit := func(docs []rag.Document) {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			if err := pipeline.Ingest(ctx, docs); err != nil {
				failedBatches.Add(1)
				log.Printf("ingest wsadu (dokumenty %d-%d) nieudany, wsadzam pojedynczo: %v", docs[0].ID, docs[len(docs)-1].ID, err)
				for _, doc := range docs {
					if err := ingestOne(doc); err != nil {
						skippedDocs.Add(1)
						log.Printf("dokument %d pominięty (zbyt długi nawet po przycięciu): %v", doc.ID, err)
					}
				}
			}
		}()
	}

	flush := func() {
		if len(batch) == 0 {
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

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		bar.Add(1)
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

		text := cl.Contents
		if len(strings.Fields(text)) > wordPrefilterThreshold {
			t, wasTruncated, err := truncate(text, *maxTokens)
			if err != nil {
				log.Printf("tokenizacja dokumentu %s nieudana, przycinam po liczbie słów: %v", cl.ID, err)
				t, wasTruncated = truncateWords(text, *maxWords)
			}
			if wasTruncated {
				text = t
				truncated.Add(1)
			}
		}

		batch = append(batch, rag.Document{ID: id, Text: text})
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

	fmt.Printf("gotowe: %d dokumentów przetworzonych do kolekcji %q w %s (przycięto do %d tokenów: %d dokumentów, nieudanych wsadów: %d, pominiętych dokumentów: %d)\n",
		total, *collection, time.Since(start).Round(time.Second), *maxTokens, truncated.Load(), failedBatches.Load(), skippedDocs.Load())
}
