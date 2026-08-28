// Command bench runs the RAG baseline (NaiveRAG) over a question set and
// measures generation quality (Exact Match, F1), retrieval quality (Recall@K,
// MRR, answer-in-context) and latency.
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
	"strings"
	"sync"
	"time"

	"github.com/mikolajsemeniuk/ragbench/pkg/flashrag"
	"github.com/mikolajsemeniuk/ragbench/pkg/metrics"
	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/rag"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// datasetLine is the FlashRAG question record; the format lives in
// pkg/flashrag so that cmd/diagnose reads the gold annotation exactly the same
// way this command scores it.
type datasetLine = flashrag.Question

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type Generator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// Pipeline is the contract every compared RAG architecture implements: given a
// question, return an answer and the passages that were in context when it was
// produced. Everything downstream - all five metrics - is derived from those
// two values, so architectures differing wildly inside are scored identically.
type Pipeline interface {
	Query(ctx context.Context, question string) (answer string, retrieved []storage.Point, err error)
}

// result holds one question's scores. Questions are evaluated concurrently, so
// results are collected by index and aggregated afterwards, keeping the run
// deterministic regardless of completion order.
type result struct {
	ok      bool
	em      float64
	f1      float64
	recall  float64
	rr      float64
	inCtx   float64
	hasGold bool
	latency time.Duration

	question string
	answer   string
	gold     []string
	passages int
}

// dumpRecord is one line of the -dump file: the per-question scores that
// aggregate means throw away.
//
// Means alone cannot answer the question a reviewer asks about any comparison
// between two architectures - "is that difference real?". A paired test
// (McNemar on Exact Match, Wilcoxon on F1) needs both systems' outcomes for
// the same questions, which only exists if each run records them. Written per
// run, consumed by cmd/compare.
type dumpRecord struct {
	Index    int      `json:"index"`
	Question string   `json:"question"`
	Answer   string   `json:"answer"`
	Gold     []string `json:"gold"`
	EM       float64  `json:"em"`
	F1       float64  `json:"f1"`
	InCtx    float64  `json:"answer_in_context"`
	HasGold  bool     `json:"has_gold"`
	Recall   float64  `json:"recall_at_k"`
	RR       float64  `json:"reciprocal_rank"`
	Passages int      `json:"passages_in_context"`
	LatencyS float64  `json:"latency_seconds"`
}

func main() {
	var (
		providerName = flag.String("provider", "vllm", "model provider: vllm | ollama")
		providerURL  = flag.String("provider-url", "http://localhost:8000", "LLM provider URL (generation)")
		embedURL     = flag.String("embed-url", "http://localhost:8001", "embedding provider URL")
		llmModel     = flag.String("llm-model", "qwen2.5-7b-instruct", "LLM model name")
		embedModel   = flag.String("embed-model", "bge-base-en-v1.5", "embedding model name")

		qdrantURL    = flag.String("qdrant-url", "http://localhost:6333", "Qdrant URL")
		collection   = flag.String("collection", "ragbench", "Qdrant collection name - it must already exist and have been filled by cmd/ingest")
		architecture = flag.String("architecture", "naive", "RAG architecture to evaluate: closedbook | naive | ircot | crag")
		ircotSteps   = flag.Int("ircot-steps", 4, "ircot only: maximum reasoning/retrieval rounds per question")
		ircotMaxDocs = flag.Int("ircot-max-passages", 15, "ircot only: cap on the accumulated passage set")
		cragMaxDocs  = flag.Int("crag-max-passages", 10, "crag only: cap on the passage set after correction")
		topK         = flag.Int("top-k", 5, "number of context passages retrieved before generation (the K in Recall@K)")

		queryPrefix = flag.String("query-prefix", rag.PrefixAuto, "instruction prepended to the question before embedding it. \"auto\" uses the convention documented for -embed-model; pass an empty string to run the no-instruction ablation, or any literal string to override. It must pair with the -doc-prefix the collection was ingested with")

		temperature = flag.Float64("temperature", 0, "generation temperature. 0 means greedy decoding, which is what makes a run reproducible; anything above it makes the reported numbers differ between runs")
		maxAnswer   = flag.Int("max-answer-tokens", 64, "hard cap on generated answer length. The gold answers are short spans, so a low cap costs nothing and bounds the runtime")

		datasetPath  = flag.String("dataset", "", "path to the FlashRAG question file (jsonl: question, golden_answers[, metadata]) - required. Evaluation always runs on a dev or test split (e.g. *_dev.jsonl, *_test.jsonl), never on *_train.jsonl: in these datasets the train split serves only as a pool of few-shot examples")
		datasetLimit = flag.Int("limit", 0, "0 = use the whole dataset; N>0 = sample only N questions (without replacement, deterministic via -seed) for a quick smoke test. Sampling rather than taking the first N lines, because some datasets order questions by type/difficulty")
		seed         = flag.Int64("seed", 42, "sampling seed for -limit (the same value yields the same, reproducible sample)")

		concurrency = flag.Int("concurrency", 1, "number of questions evaluated in parallel. Keep it at 1 when the reported latency matters: with more, the per-question timings include queueing behind other questions and are throughput, not latency. Raise it to shorten a full-split run")

		dumpPath  = flag.String("dump", "", "path of a JSONL file to write per-question scores to - required to compare two runs with cmd/compare, since paired significance tests need per-question outcomes")
		resamples = flag.Int("bootstrap", 10000, "bootstrap resamples for the 95% confidence intervals; 0 disables them")
		texOut    = flag.String("tex-out", "", "path of the .tex file to generate with the results (e.g. paper/baseline.gen.tex) - optional")
		name      = flag.String("name", "NaiveRAG", "baseline name used in the generated .tex")
	)
	flag.Parse()

	if *datasetPath == "" {
		log.Fatal("missing required flag -dataset")
	}
	if strings.Contains(filepath.Base(*datasetPath), "_train") {
		log.Printf("WARNING: -dataset points at a train split (%s) - QA/RAG evaluation canonically runs on dev/test, train is only for few-shot examples", *datasetPath)
	}
	if *concurrency < 1 {
		log.Fatalf("-concurrency must be >= 1, got %d", *concurrency)
	}

	ctx := context.Background()

	var embedder Embedder
	var generator Generator
	switch *providerName {
	case "vllm":
		embedder = provider.NewVLLM(*embedURL, *embedModel)
		gen := provider.NewVLLM(*providerURL, *llmModel)
		gen.Temperature = *temperature
		gen.MaxTokens = *maxAnswer
		generator = gen
	case "ollama":
		embedder = provider.NewOllama(*embedURL, *embedModel)
		gen := provider.NewOllama(*providerURL, *llmModel)
		gen.Temperature = *temperature
		gen.MaxTokens = *maxAnswer
		generator = gen
	default:
		log.Fatalf("unknown provider: %s (expected vllm | ollama)", *providerName)
	}

	store := storage.NewQdrant(*qdrantURL)
	resolvedPrefix := rag.ResolvePrefix(*queryPrefix, *embedModel, true)

	var pipeline Pipeline
	switch *architecture {
	case "closedbook":
		pipeline = rag.NewClosedBook(generator)
		log.Printf("architecture: closedbook (no retrieval - measures what the generator knows on its own)")
	case "naive":
		p := rag.NewNaiveRAG(store, *collection, embedder, generator)
		p.TopK = *topK
		p.QueryPrefix = resolvedPrefix
		pipeline = p
	case "ircot":
		p := rag.NewIRCoT(store, *collection, embedder, generator)
		p.TopK = *topK
		p.MaxSteps = *ircotSteps
		p.MaxPassages = *ircotMaxDocs
		p.QueryPrefix = resolvedPrefix
		pipeline = p
		log.Printf("architecture: ircot (max %d reasoning rounds, up to %d passages accumulated)", *ircotSteps, *ircotMaxDocs)
	case "crag":
		p := rag.NewCRAG(store, *collection, embedder, generator)
		p.TopK = *topK
		p.MaxPassages = *cragMaxDocs
		p.QueryPrefix = resolvedPrefix
		pipeline = p
		log.Printf("architecture: crag (grade retrieval, rewrite and re-search when it is poor, up to %d passages)", *cragMaxDocs)
	default:
		log.Fatalf("unknown architecture: %s (expected closedbook | naive | ircot | crag)", *architecture)
	}

	// The query instruction materially changes retrieval, so it is reported
	// rather than applied silently - a run's numbers are only comparable with
	// another run that used the same convention.
	if _, _, known := rag.PrefixesFor(*embedModel); !known && *queryPrefix == rag.PrefixAuto {
		log.Printf("query prefix: none (no convention known for embedding model %q; pass -query-prefix explicitly if it expects one)", *embedModel)
	} else {
		log.Printf("query prefix: %q", resolvedPrefix)
	}

	dataset, err := loadDataset(*datasetPath)
	if err != nil {
		log.Fatalf("loading the question set: %v", err)
	}
	if *datasetLimit > 0 && *datasetLimit < len(dataset) {
		rng := rand.New(rand.NewSource(*seed))
		total := len(dataset)
		rng.Shuffle(total, func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })
		dataset = dataset[:*datasetLimit]
		log.Printf("smoke test: sampled %d/%d questions (seed=%d)", len(dataset), total, *seed)
	}

	results := make([]result, len(dataset))
	var done, failed int64
	var mu sync.Mutex

	start := time.Now()
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range *concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = evaluate(ctx, pipeline, dataset[i], *topK)
				results[i].question = dataset[i].Question
				results[i].gold = dataset[i].GoldenAnswers
				mu.Lock()
				done++
				if !results[i].ok {
					failed++
				}
				if done%200 == 0 || int(done) == len(dataset) {
					log.Printf("%d/%d questions evaluated (%s elapsed)", done, len(dataset), time.Since(start).Round(time.Second))
				}
				mu.Unlock()
			}
		}()
	}
	for i := range dataset {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	wall := time.Since(start)

	var sumEM, sumF1, sumRecall, sumRR, sumInCtx, sumPassages float64
	var sumLatency time.Duration
	var answered, withGold int
	for _, r := range results {
		if !r.ok {
			continue
		}
		answered++
		sumEM += r.em
		sumF1 += r.f1
		sumInCtx += r.inCtx
		sumPassages += float64(r.passages)
		sumLatency += r.latency
		if r.hasGold {
			withGold++
			sumRecall += r.recall
			sumRR += r.rr
		}
	}
	if answered == 0 {
		log.Fatalf("every question failed (%d errors) - check that the collection %q exists and both model servers are up", failed, *collection)
	}

	n := float64(answered)
	em := sumEM / n
	f1 := sumF1 / n
	inCtx := sumInCtx / n
	avgLatency := time.Duration(float64(sumLatency) / n)

	var recall, mrr float64
	if withGold > 0 {
		recall = sumRecall / float64(withGold)
		mrr = sumRR / float64(withGold)
	}

	// Collect the per-question values the intervals and the dump are built
	// from. Only answered questions contribute; a failed query is excluded
	// rather than scored as zero, so an outage cannot masquerade as a bad
	// result.
	var emVals, f1Vals, inCtxVals, recallVals, rrVals []float64
	for _, r := range results {
		if !r.ok {
			continue
		}
		emVals = append(emVals, r.em)
		f1Vals = append(f1Vals, r.f1)
		inCtxVals = append(inCtxVals, r.inCtx)
		if r.hasGold {
			recallVals = append(recallVals, r.recall)
			rrVals = append(rrVals, r.rr)
		}
	}

	ci := func(values []float64) string {
		if *resamples <= 0 || len(values) == 0 {
			return ""
		}
		lo, hi := metrics.BootstrapCI(values, *resamples, rand.New(rand.NewSource(*seed)))
		return fmt.Sprintf("  95%% CI [%.4f, %.4f]", lo, hi)
	}

	if *dumpPath != "" {
		if err := writeDump(*dumpPath, results); err != nil {
			log.Fatalf("writing the dump file: %v", err)
		}
		log.Printf("per-question scores written to %s", *dumpPath)
	}

	fmt.Println("--- results ---")
	fmt.Printf("dataset:            %s\n", *datasetPath)
	fmt.Printf("collection:         %s (top-k=%d)\n", *collection, *topK)
	fmt.Printf("architecture:       %s\n", *architecture)
	fmt.Printf("embedding model:    %s (query prefix %q)\n", *embedModel, resolvedPrefix)
	fmt.Printf("llm:                %s (temperature=%g, max answer tokens=%d)\n", *llmModel, *temperature, *maxAnswer)
	fmt.Printf("questions:          %d evaluated", answered)
	if failed > 0 {
		fmt.Printf(", %d failed", failed)
	}
	fmt.Println()
	fmt.Printf("Exact Match:        %.4f%s\n", em, ci(emVals))
	fmt.Printf("F1:                 %.4f%s\n", f1, ci(f1Vals))
	fmt.Printf("Answer in context:  %.4f%s\n", inCtx, ci(inCtxVals))
	fmt.Printf("Passages in ctx:    %.1f on average\n", sumPassages/n)
	if withGold > 0 {
		fmt.Printf("Recall (context):   %.4f%s (gold articles among the passages in context, over %d/%d annotated questions)\n", recall, ci(recallVals), withGold, answered)
		fmt.Printf("MRR:                %.4f%s\n", mrr, ci(rrVals))
	} else {
		fmt.Printf("Recall@%d/MRR:       not computed (this dataset annotates no gold documents)\n", *topK)
	}
	fmt.Printf("wall clock:         %s (%.1f questions/s, concurrency=%d)\n", wall.Round(time.Second), n/wall.Seconds(), *concurrency)
	if *concurrency == 1 {
		fmt.Printf("Avg latency:        %s\n", avgLatency.Round(time.Millisecond))
	} else {
		fmt.Printf("Avg latency:        %s - NOT a latency measurement at concurrency=%d, it includes queueing. Re-run with -concurrency 1 on a subsample to report latency.\n", avgLatency.Round(time.Millisecond), *concurrency)
	}

	if *texOut != "" {
		if err := writeTex(*texOut, *name, answered, em, f1, inCtx, avgLatency, *topK, recall, mrr, withGold); err != nil {
			log.Fatalf("writing the tex file: %v", err)
		}
		log.Printf("results written to %s", *texOut)
	}
}

// evaluate runs one question through the pipeline and scores it.
//
// Retrieval is scored at the article level: a Wikipedia article is split into
// many ~100-word passages sharing one title, and the datasets annotate gold
// *articles*, not passages. Counting passages instead would cap Recall@5 at
// roughly 0.31 on HotpotQA - measured - producing a number that cannot be
// compared with any published Recall@5.
func evaluate(ctx context.Context, pipeline Pipeline, item datasetLine, topK int) result {
	start := time.Now()
	answer, passages, err := pipeline.Query(ctx, item.Question)
	latency := time.Since(start)
	if err != nil {
		log.Printf("question %q: query failed: %v", item.Question, err)
		return result{}
	}

	texts := make([]string, len(passages))
	titles := make([]string, len(passages))
	for i, p := range passages {
		texts[i] = p.Text
		titles[i] = flashrag.PassageTitle(p.Text)
	}

	r := result{
		ok:       true,
		answer:   answer,
		passages: len(passages),
		em:       metrics.ExactMatch(answer, item.GoldenAnswers),
		f1:       metrics.F1Score(answer, item.GoldenAnswers),
		inCtx:    metrics.AnswerInContext(texts, item.GoldenAnswers),
		latency:  latency,
	}
	if gold := flashrag.GoldTitles(item.Metadata); len(gold) > 0 {
		r.hasGold = true
		// Scored over every passage placed in the generator's context, not
		// over the first topK. An architecture that accumulates passages
		// across several retrieval rounds (IRCoT) puts more than topK in
		// context; trimming to topK would discard exactly the passages its
		// extra rounds fetched and report it as identical to a single-query
		// baseline. For a single-query architecture the two are the same
		// thing, so this leaves NaiveRAG's numbers unchanged.
		r.recall = metrics.RecallAtK(titles, gold, len(titles))
		r.rr = metrics.ReciprocalRank(titles, gold)
	}
	return r
}

// writeTex writes the results as a LaTeX fragment (\newcommand definitions),
// ready to \input{} from the body of the paper, e.g.:
//
//	\input{baseline.gen.tex}
//	Exact Match: \NaiveRAGEM, F1: \NaiveRAGFOne
func writeTex(path, name string, n int, em, f1, inCtx float64, avgLatency time.Duration, topK int, recall, mrr float64, withGold int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}

	id := texSafeID(name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/bench - do not edit by hand\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sQuestions}{%d}\n", id, n)
	fmt.Fprintf(&b, "\\newcommand{\\%sEM}{%.4f}\n", id, em)
	fmt.Fprintf(&b, "\\newcommand{\\%sFOne}{%.4f}\n", id, f1)
	fmt.Fprintf(&b, "\\newcommand{\\%sAnswerInContext}{%.4f}\n", id, inCtx)
	fmt.Fprintf(&b, "\\newcommand{\\%sLatency}{%s}\n", id, avgLatency.Round(time.Millisecond))
	if withGold > 0 {
		fmt.Fprintf(&b, "\\newcommand{\\%sRecallAtK}{%.4f}\n", id, recall)
		fmt.Fprintf(&b, "\\newcommand{\\%sMRR}{%.4f}\n", id, mrr)
		fmt.Fprintf(&b, "\\newcommand{\\%sTopK}{%d}\n", id, topK)
		fmt.Fprintf(&b, "\\newcommand{\\%sRetrievalEvalQuestions}{%d}\n", id, withGold)
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

// writeDump records the per-question scores as JSONL, one object per line, in
// dataset order.
func writeDump(path string, results []result) error {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create dir: %w", err)
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, r := range results {
		if !r.ok {
			continue
		}
		if err := enc.Encode(dumpRecord{
			Index:    i,
			Question: r.question,
			Answer:   r.answer,
			Gold:     r.gold,
			EM:       r.em,
			F1:       r.f1,
			InCtx:    r.inCtx,
			HasGold:  r.hasGold,
			Recall:   r.recall,
			RR:       r.rr,
			Passages: r.passages,
			LatencyS: r.latency.Seconds(),
		}); err != nil {
			return err
		}
	}
	return nil
}
