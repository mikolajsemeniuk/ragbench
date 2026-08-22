// Command bench runs the RAG baseline (NaiveRAG) over a question set and
// measures generation quality (Exact Match, F1), retrieval quality (Recall@K,
// MRR - only for datasets that annotate gold documents, e.g. HotpotQA,
// 2WikiMultihopQA, MuSiQue) and latency.
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

// corpusLine mirrors the FlashRAG corpus format (e.g. wiki18_100w.jsonl):
// {"id": "0", "contents": "\"Title\"\nPassage text..."}
type corpusLine struct {
	ID       string `json:"id"`
	Contents string `json:"contents"`
}

// datasetLine mirrors the FlashRAG question-set format. Metadata is kept raw
// because its shape differs between datasets - see extractGoldTitles.
type datasetLine struct {
	Question      string          `json:"question"`
	GoldenAnswers []string        `json:"golden_answers"`
	Metadata      json.RawMessage `json:"metadata"`
}

// datasetMetadata covers the two shapes of gold-document annotation found in
// the FlashRAG datasets:
//   - HotpotQA / 2WikiMultihopQA: metadata.supporting_facts.title
//   - MuSiQue: metadata.question_decomposition[].support_paragraph.title
//
// NaturalQuestions and TriviaQA carry no such annotation (open-domain QA with
// no designated gold passages), so Recall@K/MRR cannot be computed for them.
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

// extractGoldTitles returns the unique gold document titles for a question if
// the dataset annotates them, and nil otherwise.
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
		providerName = flag.String("provider", "vllm", "model provider: vllm | ollama")
		providerURL  = flag.String("provider-url", "http://localhost:8000", "LLM provider URL (generation)")
		embedURL     = flag.String("embed-url", "http://localhost:8001", "embedding provider URL")
		llmModel     = flag.String("llm-model", "qwen2.5-7b-instruct", "LLM model name")
		embedModel   = flag.String("embed-model", "bge-base-en-v1.5", "embedding model name")

		qdrantURL  = flag.String("qdrant-url", "http://localhost:6333", "Qdrant URL")
		collection = flag.String("collection", "ragbench", "Qdrant collection name - it must already exist and have been filled by cmd/ingest")
		topK       = flag.Int("top-k", 5, "number of context passages retrieved before generation (the K in Recall@K)")

		queryPrefix = flag.String("query-prefix", rag.PrefixAuto, "instruction prepended to the question before embedding it. \"auto\" uses the convention documented for -embed-model (BGE English: \"Represent this sentence for searching relevant passages: \"); pass an empty string to run the no-instruction ablation, or any literal string to override. It must pair with the -doc-prefix the collection was ingested with; because it only affects the query embedding it can be changed without re-ingesting")

		corpusPath   = flag.String("corpus", "", "path to the FlashRAG corpus file (jsonl: id, contents) the collection was ingested from (see cmd/ingest) - optional; it ingests nothing, it only builds the title->ID index needed to compute Recall@K/MRR")
		corpusLimit  = flag.Int("corpus-limit", 0, "0 = build the title->ID index from the whole corpus; N>0 = sample only N documents into the index (reservoir sampling, deterministic via -seed) for a quick smoke test. NOTE: with a small N the gold documents will most likely miss the sample, so Recall@K/MRR come out understated and unrepresentative - this is a 'does it run' check, not a quality benchmark")
		datasetPath  = flag.String("dataset", "", "path to the FlashRAG question file (jsonl: question, golden_answers[, metadata]) - required. Evaluation always runs on a dev or test split (e.g. *_dev.jsonl, *_test.jsonl), never on *_train.jsonl: in these datasets the train split serves only as a pool of few-shot examples, as in FlashRAG/Self-RAG/IRCoT/Adaptive-RAG and the other works listed in the README")
		datasetLimit = flag.Int("limit", 0, "0 = use the whole dataset; N>0 = sample only N questions (without replacement, deterministic via -seed) for a quick smoke test. Sampling rather than taking the first N lines, because some datasets (e.g. HotpotQA) order questions by type/difficulty, so the first N would be an unrepresentative, skewed sample")
		seed         = flag.Int64("seed", 42, "sampling seed for -limit/-corpus-limit (the same value yields the same, reproducible sample)")
		texOut       = flag.String("tex-out", "", "path of the .tex file to generate with the results (e.g. paper/baseline.gen.tex) - optional")
		name         = flag.String("name", "NaiveRAG", "baseline name used in the generated .tex")
	)
	flag.Parse()

	if *datasetPath == "" {
		log.Fatal("missing required flag -dataset")
	}
	if strings.Contains(filepath.Base(*datasetPath), "_train") {
		log.Printf("WARNING: -dataset points at a train split (%s) - QA/RAG evaluation canonically runs on dev/test, train is only for few-shot examples", *datasetPath)
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
		log.Fatalf("unknown provider: %s (expected vllm | ollama)", *providerName)
	}

	store := storage.NewQdrant(*qdrantURL)
	pipeline := rag.NewNaiveRAG(store, *collection, embedder, generator)
	pipeline.TopK = *topK
	pipeline.QueryPrefix = rag.ResolvePrefix(*queryPrefix, *embedModel, true)

	// The query instruction materially changes retrieval quality, so it is
	// reported rather than applied silently - a run's numbers are only
	// comparable with another run that used the same convention.
	if _, _, known := rag.PrefixesFor(*embedModel); !known && *queryPrefix == rag.PrefixAuto {
		log.Printf("query prefix: none (no convention known for embedding model %q; pass -query-prefix explicitly if it expects one)", *embedModel)
	} else {
		log.Printf("query prefix: %q", pipeline.QueryPrefix)
	}

	rng := rand.New(rand.NewSource(*seed))

	// titleIndex maps an article title to the IDs of all its passages in the
	// corpus. It is what turns the gold titles annotated in the dataset into
	// the gold document IDs that Recall@K/MRR are computed against. cmd/bench
	// ingests nothing itself - the collection must already have been filled by
	// cmd/ingest.
	var titleIndex map[string][]uint64
	if *corpusPath != "" {
		index, err := buildTitleIndex(*corpusPath, *corpusLimit, rng)
		if err != nil {
			log.Fatalf("building the title->ID index: %v", err)
		}
		if *corpusLimit > 0 {
			log.Printf("smoke test: title->ID index built from a random sample of %d documents (seed=%d) instead of the whole corpus", *corpusLimit, *seed)
		}
		titleIndex = index
	} else {
		log.Printf("no -corpus given: Recall@K/MRR will not be computed (no title->ID index); only EM/F1/latency will be")
	}

	dataset, err := loadDataset(*datasetPath)
	if err != nil {
		log.Fatalf("loading the question set: %v", err)
	}
	if *datasetLimit > 0 && *datasetLimit < len(dataset) {
		total := len(dataset)
		rng.Shuffle(total, func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })
		dataset = dataset[:*datasetLimit]
		log.Printf("smoke test: sampled %d/%d questions (seed=%d)", len(dataset), total, *seed)
	}

	var sumEM, sumF1, sumRecall, sumRR float64
	var sumLatency time.Duration
	var retrievalEvalCount int
	for i, item := range dataset {
		start := time.Now()
		answer, retrievedIDs, err := pipeline.Query(ctx, item.Question)
		latency := time.Since(start)
		if err != nil {
			log.Printf("question %d (%q): query failed: %v", i, item.Question, err)
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

	fmt.Println("--- baseline results (NaiveRAG) ---")
	fmt.Printf("questions:       %d\n", len(dataset))
	fmt.Printf("query prefix:    %q\n", pipeline.QueryPrefix)
	fmt.Printf("Exact Match:     %.4f\n", em)
	fmt.Printf("F1:              %.4f\n", f1)
	fmt.Printf("Avg latency:     %s\n", avgLatency)

	var recall, mrr float64
	if retrievalEvalCount > 0 {
		recall = sumRecall / float64(retrievalEvalCount)
		mrr = sumRR / float64(retrievalEvalCount)
		fmt.Printf("Recall@%d:        %.4f (over %d/%d questions with gold documents)\n", pipeline.TopK, recall, retrievalEvalCount, len(dataset))
		fmt.Printf("MRR:             %.4f (over %d/%d questions with gold documents)\n", mrr, retrievalEvalCount, len(dataset))
	}

	if *texOut != "" {
		if err := writeTex(*texOut, *name, len(dataset), em, f1, avgLatency, pipeline.TopK, recall, mrr, retrievalEvalCount); err != nil {
			log.Fatalf("writing the tex file: %v", err)
		}
		log.Printf("results written to %s", *texOut)
	}
}

// goldRelevantIDs turns the gold document titles recorded in a question's
// metadata into corpus passage IDs, via the titleIndex. It returns nil if the
// dataset annotates no gold documents, or if no corpus was supplied to this
// run.
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

// writeTex writes the results as a LaTeX fragment (\newcommand definitions),
// ready to \input{} from the body of the paper, e.g.:
//
//	\input{baseline.gen.tex}
//	Exact Match: \NaiveRAGEM, F1: \NaiveRAGFOne
func writeTex(path, name string, n int, em, f1 float64, avgLatency time.Duration, topK int, recall, mrr float64, retrievalEvalCount int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}

	id := texSafeID(name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/bench - do not edit by hand\n")
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

// texSafeID drops every character outside [A-Za-z], because a LaTeX
// \newcommand name may contain letters only.
func texSafeID(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// extractTitle pulls the article title out of the FlashRAG contents field,
// whose first line is the quoted title, e.g. "\"Title\"\nText...".
func extractTitle(contents string) string {
	line, _, _ := strings.Cut(contents, "\n")
	return strings.Trim(line, "\"")
}

// titleEntry is one entry of the title->ID index built by buildTitleIndex.
// Only the title and the passage ID are kept, never the full text, so that
// indexing an entire corpus (e.g. the 21M passages of wiki18_100w.jsonl) does
// not require holding the passage contents in memory.
type titleEntry struct {
	id    uint64
	title string
}

// buildTitleIndex reads a FlashRAG corpus and builds the title->passage-ID
// index needed to resolve gold document IDs when computing Recall@K/MRR (one
// Wikipedia article is split into many ~100-word passages that all share a
// title). The corpus itself is not ingested into any vector database here -
// that is cmd/ingest's job.
//
// If limit > 0, reservoir sampling is used instead of reading the whole file
// (21M lines / 14GB for wiki18_100w.jsonl): every line has an equal chance of
// entering the sample, and at most `limit` entries are held in memory
// regardless of file size.
func buildTitleIndex(path string, limit int, rng *rand.Rand) (map[string][]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []titleEntry
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
			return nil, fmt.Errorf("parse corpus line: %w", err)
		}
		id, err := strconv.ParseUint(cl.ID, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse corpus id %q: %w", cl.ID, err)
		}
		entry := titleEntry{id: id, title: extractTitle(cl.Contents)}

		if limit <= 0 {
			entries = append(entries, entry)
			continue
		}

		// Algorithm R (reservoir sampling).
		if len(entries) < limit {
			entries = append(entries, entry)
		} else if j := rng.Intn(seen + 1); j < limit {
			entries[j] = entry
		}
		seen++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	titleIndex := make(map[string][]uint64, len(entries))
	for _, e := range entries {
		if e.title != "" {
			titleIndex[e.title] = append(titleIndex[e.title], e.id)
		}
	}
	return titleIndex, nil
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
