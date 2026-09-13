// Command bench runs one RAG architecture over a question set and measures
// generation quality (Exact Match, F1, abstention), retrieval quality
// (Recall, MRR, answer-in-context) and cost (generation calls, tokens,
// latency).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"maps"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
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
// produced. Everything downstream is derived from those two values, so
// architectures differing wildly inside are scored identically.
type Pipeline = rag.Pipeline

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
	latency time.Duration

	// hasGold gates Recall and MRR to the questions whose dataset annotates
	// gold documents; inCtxOK gates answer-in-context to the questions where
	// that metric means anything at all - see metrics.AnswerInContextApplicable.
	hasGold bool
	inCtxOK bool

	// abstained marks an answer that declined rather than answered. Scored
	// separately because Exact Match cannot tell a refusal from a wrong guess
	// and the two say opposite things about a system.
	abstained bool

	question        string
	answer          string
	gold            []string
	passages        int
	retrievedIDs    []uint64
	retrievedTitles []string

	promptTokens     int64
	completionTokens int64
	llmCalls         int64

	trace rag.TraceData

	meanLogprob *float64
}

// dumpRecord is one line of the -dump file: the per-question detail that
// aggregate means throw away.
//
// Two things are recorded that a scores-only dump would not. First, both
// systems' outcomes for the same questions, which is what a paired test
// (McNemar on Exact Match, Wilcoxon on F1) needs - consumed by cmd/compare.
// Second, what was actually retrieved and what the architecture decided along
// the way. Without the retrieved ids and titles, changing a retrieval metric
// means re-running the whole split on the GPU instead of recomputing it from
// the file, and no reviewer can check the retrieval numbers at all.
type dumpRecord struct {
	Index    int      `json:"index"`
	Question string   `json:"question"`
	Answer   string   `json:"answer"`
	Gold     []string `json:"gold"`

	EM        float64 `json:"em"`
	F1        float64 `json:"f1"`
	Abstained bool    `json:"abstained"`

	InCtx           float64 `json:"answer_in_context"`
	InCtxApplicable bool    `json:"answer_in_context_applicable"`
	HasGold         bool    `json:"has_gold"`
	Recall          float64 `json:"recall_in_context"`
	RR              float64 `json:"reciprocal_rank"`

	Passages        int      `json:"passages_in_context"`
	RetrievedIDs    []uint64 `json:"retrieved_ids"`
	RetrievedTitles []string `json:"retrieved_titles"`

	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	LLMCalls         int64 `json:"llm_calls"`

	// The architecture's own decisions, present only for the architectures
	// that make them.
	Route          string   `json:"route,omitempty"`
	Branch         string   `json:"branch,omitempty"`
	GradedKept     int      `json:"graded_kept,omitempty"`
	RewrittenQuery string   `json:"rewritten_query,omitempty"`
	ReasoningSteps int      `json:"reasoning_steps,omitempty"`
	Stage          string   `json:"stage,omitempty"`
	StagesRun      int      `json:"stages_run,omitempty"`
	Escalations    []string `json:"escalations,omitempty"`

	// MeanLogprob is the generator's confidence in the final answer: the mean
	// log-probability of its tokens. Absent when the provider reports none.
	// It is recorded for every architecture so that a confidence threshold
	// can be swept offline (cmd/sweep) instead of re-running the GPU.
	MeanLogprob *float64 `json:"mean_logprob,omitempty"`

	LatencyS float64 `json:"latency_seconds"`
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
		architecture = flag.String("architecture", "naive", "RAG architecture to evaluate: closedbook | naive | ircot | crag | rerank | hyde | bm25 | hybrid | fused | adaptive | neighbour | cascade")
		ircotSteps   = flag.Int("ircot-steps", 4, "ircot only: maximum reasoning/retrieval rounds per question")
		ircotMaxDocs = flag.Int("ircot-max-passages", 15, "ircot only: cap on the accumulated passage set")
		ircotDemo    = flag.Bool("ircot-demo", false, "ircot only: prepend one worked example to the reasoning prompt, as the original method and FlashRAG do. Off by default so that the zero-shot runs stay reproducible; the one-shot run is a separate row")
		cragMaxDocs  = flag.Int("crag-max-passages", 10, "crag only: cap on the passage set after correction")
		rerankURL    = flag.String("rerank-url", "http://localhost:8002", "rerank only: cross-encoder provider URL")
		rerankModel  = flag.String("rerank-model", "bge-reranker-base", "rerank only: cross-encoder model name")
		candidates   = flag.Int("candidates", 100, "rerank/bm25/hybrid: how deep each retriever's shortlist goes before it is reordered or fused. The measured payoff comes from candidates the top-k cut discards, so this has to exceed -top-k by a wide margin")
		topK         = flag.Int("top-k", 5, "number of context passages retrieved before generation")

		hnswEf = flag.Int("hnsw-ef", 256, "size of the candidate list Qdrant keeps while walking the HNSW graph. 0 leaves the server's default, which makes the approximate search's own recall an unreported property of the run. It has to be at least -candidates for a deep shortlist to be meaningful")

		sparseColl    = flag.String("sparse-collection", "ragbench-wiki18-bm25", "bm25/hybrid only: Qdrant collection holding the lexical index - it must already exist and have been filled by cmd/ingest -sparse")
		cascadeMinLP  = flag.Float64("cascade-min-logprob", math.Inf(-1), "cascade only: also escalate an answer whose mean token log-probability is below this. -Inf (the default) leaves the lossless abstention trigger alone. Choose the value with cmd/sweep on the train splits, never on the test sets the table reports")
		cascadeStages = flag.String("cascade-stages", "fused,hyde,closedbook", "cascade only: comma-separated architectures to try in order. A stage's answer is kept unless it declines to answer, in which case the next stage runs. The refusal is a posterior signal - the reader has already seen the retrieved passages - and it is lossless under Exact Match, so escalating cannot discard a correct answer")

		rrfK = flag.Int("rrf-k", 60, "hybrid only: the k of Reciprocal Rank Fusion. Larger values flatten the influence of the top positions; 60 is the value the RRF paper proposes and what every implementation uses")

		neighbourRadius  = flag.Int("neighbour-radius", 1, "neighbour only: how many passages to either side of each hit to pull in from the same article")
		neighbourMaxDocs = flag.Int("neighbour-max-passages", 15, "neighbour only: cap on the expanded passage set. Compare the run against a naive run with a matching -top-k, not against the 5-passage baseline")
		titleIndexPath   = flag.String("title-index", "dataset/wiki18_100w.titles.gob", "neighbour only: path of the article-title index. Built from -corpus on first use and reused afterwards, because the corpus is not laid out in article order and the neighbours of a passage cannot be derived from its id")
		corpusPath       = flag.String("corpus", "dataset/wiki18_100w.jsonl", "neighbour only: FlashRAG corpus the collection was built from, used to build -title-index when it does not exist yet")

		queryPrefix = flag.String("query-prefix", rag.PrefixAuto, "instruction prepended to the question before embedding it. \"auto\" uses the convention documented for -embed-model; pass an empty string to run the no-instruction ablation, or any literal string to override. It must pair with the -doc-prefix the collection was ingested with")

		temperature = flag.Float64("temperature", 0, "generation temperature. 0 means greedy decoding, which is what makes a run reproducible; anything above it makes the reported numbers differ between runs")
		maxAnswer   = flag.Int("max-answer-tokens", 64, "hard cap on generated answer length. The gold answers are short spans, so a low cap costs nothing and bounds the runtime")
		maxStep     = flag.Int("max-step-tokens", 256, "hard cap on an intermediate generation call that is not the answer: an IRCoT reasoning sentence, a CRAG query rewrite, a HyDE draft. It needs a larger budget than the answer - a reasoning sentence truncated at the answer's cap is half a sentence, and it is then used verbatim as the next retrieval query")

		abstentionWords = flag.Int("abstention-max-words", metrics.DefaultAbstentionMaxWords, "an answer longer than this counts as a refusal rather than an answer. 0 disables the length signal and leaves only the explicit refusal phrases")

		datasetPath  = flag.String("dataset", "", "path to the FlashRAG question file (jsonl: question, golden_answers[, metadata]) - required. Evaluation always runs on a dev or test split (e.g. *_dev.jsonl, *_test.jsonl), never on *_train.jsonl: in these datasets the train split serves only as a pool of few-shot examples")
		datasetLimit = flag.Int("limit", 0, "0 = use the whole dataset; N>0 = sample only N questions (without replacement, deterministic via -seed) for a quick smoke test. Sampling rather than taking the first N lines, because some datasets order questions by type/difficulty")
		seed         = flag.Int64("seed", 42, "sampling seed for -limit (the same value yields the same, reproducible sample)")

		concurrency = flag.Int("concurrency", 1, "number of questions evaluated in parallel. Keep it at 1 when the reported latency matters: with more, the per-question timings include queueing behind other questions and are throughput, not latency. Raise it to shorten a full-split run")

		dumpPath  = flag.String("dump", "", "path of a JSONL file to write per-question detail to - required to compare two runs with cmd/compare, and the only way to recompute a retrieval metric without re-running the split")
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
	if *hnswEf > 0 && *hnswEf < *candidates {
		log.Printf("WARNING: -hnsw-ef %d is below -candidates %d, so the shortlist is limited by the graph walk rather than by the ranking", *hnswEf, *candidates)
	}

	ctx := context.Background()

	// generator produces the final answer and is capped accordingly; drafter
	// is the same model on the same server with a larger token budget, used
	// for the intermediate calls whose output is prose rather than a span.
	var embedder Embedder
	var generator, drafter Generator
	switch *providerName {
	case "vllm":
		embedder = provider.NewVLLM(*embedURL, *embedModel)
		gen := provider.NewVLLM(*providerURL, *llmModel)
		gen.Temperature = *temperature
		gen.MaxTokens = *maxAnswer
		generator = gen
		draft := provider.NewVLLM(*providerURL, *llmModel)
		draft.Temperature = *temperature
		draft.MaxTokens = *maxStep
		drafter = draft
	case "ollama":
		embedder = provider.NewOllama(*embedURL, *embedModel)
		gen := provider.NewOllama(*providerURL, *llmModel)
		gen.Temperature = *temperature
		gen.MaxTokens = *maxAnswer
		generator = gen
		draft := provider.NewOllama(*providerURL, *llmModel)
		draft.Temperature = *temperature
		draft.MaxTokens = *maxStep
		drafter = draft
	default:
		log.Fatalf("unknown provider: %s (expected vllm | ollama)", *providerName)
	}

	store := storage.NewQdrant(*qdrantURL)
	store.HNSWEf = *hnswEf
	resolvedPrefix := rag.ResolvePrefix(*queryPrefix, *embedModel, true)

	// newNaive and newIRCoT are shared with the adaptive router, which is
	// built out of the other architectures rather than reimplementing them -
	// that is what keeps its rows comparable with theirs.
	newNaive := func() *rag.NaiveRAG {
		p := rag.NewNaiveRAG(store, *collection, embedder, generator)
		p.TopK = *topK
		p.QueryPrefix = resolvedPrefix
		return p
	}
	newIRCoT := func() *rag.IRCoT {
		p := rag.NewIRCoT(store, *collection, embedder, generator)
		p.Stepper = drafter
		p.TopK = *topK
		p.MaxSteps = *ircotSteps
		p.MaxPassages = *ircotMaxDocs
		if *ircotDemo {
			p.Demonstration = rag.IRCoTDemonstration
		}
		p.QueryPrefix = resolvedPrefix
		return p
	}

	// build is recursive because the cascade is assembled from the other
	// architectures rather than reimplementing them, which is what keeps its
	// stages comparable with the rows they appear next to in the table.
	var build func(string) Pipeline
	build = func(arch string) Pipeline {
		switch arch {
		case "closedbook":
			return rag.NewClosedBook(generator)
		case "naive":
			return newNaive()
		case "hyde":
			p := rag.NewHyDE(store, *collection, embedder, generator, drafter)
			p.TopK = *topK
			p.QueryPrefix = resolvedPrefix
			p.DocumentPrefix = rag.ResolvePrefix(rag.PrefixAuto, *embedModel, false)
			return p
		case "bm25":
			p := rag.NewBM25RAG(store, *sparseColl, generator)
			p.TopK = *topK
			p.Candidates = *candidates
			return p
		case "hybrid":
			p := rag.NewHybridRAG(store, store, *collection, *sparseColl, embedder, generator)
			p.TopK = *topK
			p.Candidates = *candidates
			p.RRFK = *rrfK
			p.QueryPrefix = resolvedPrefix
			return p
		case "fused":
			rr := provider.NewVLLM(*rerankURL, *rerankModel)
			p := rag.NewFusedRAG(store, store, *collection, *sparseColl, embedder, rr, generator)
			p.TopK = *topK
			p.Candidates = *candidates
			p.RRFK = *rrfK
			p.QueryPrefix = resolvedPrefix
			p.DocumentPrefix = rag.ResolvePrefix(rag.PrefixAuto, *embedModel, false)
			return p
		case "neighbour":
			started := time.Now()
			index, err := flashrag.OpenTitleIndex(*titleIndexPath, *corpusPath)
			if err != nil {
				log.Fatalf("title index: %v", err)
			}
			log.Printf("title index: %d articles, %d passages (%s)", len(index.ByTitle), index.Passages(), time.Since(started).Round(time.Second))
			p := rag.NewNeighbourRAG(store, store, *collection, embedder, generator, index)
			p.TopK = *topK
			p.Radius = *neighbourRadius
			p.MaxPassages = *neighbourMaxDocs
			p.QueryPrefix = resolvedPrefix
			return p
		case "adaptive":
			return rag.NewAdaptiveRAG(generator, rag.NewClosedBook(generator), newNaive(), newIRCoT())
		case "ircot":
			return newIRCoT()
		case "rerank":
			rr := provider.NewVLLM(*rerankURL, *rerankModel)
			p := rag.NewRerankRAG(store, *collection, embedder, rr, generator)
			p.TopK = *topK
			p.Candidates = *candidates
			p.QueryPrefix = resolvedPrefix
			return p
		case "crag":
			p := rag.NewCRAG(store, *collection, embedder, generator)
			p.Stepper = drafter
			p.TopK = *topK
			p.MaxPassages = *cragMaxDocs
			p.QueryPrefix = resolvedPrefix
			return p
		case "cascade":
			names := strings.Split(*cascadeStages, ",")
			stages := make([]Pipeline, 0, len(names))
			for i, n := range names {
				names[i] = strings.TrimSpace(n)
				if names[i] == "cascade" {
					log.Fatalf("-cascade-stages may not contain \"cascade\"")
				}
				stages = append(stages, build(names[i]))
			}
			log.Printf("architecture: cascade (%s; a stage answers unless it declines, in which case the next one runs)", strings.Join(names, " -> "))
			c := rag.NewCascadeRAG(names, stages, *abstentionWords)
			c.MinLogprob = *cascadeMinLP
			if !math.IsInf(*cascadeMinLP, -1) {
				log.Printf("cascade: also escalating answers with mean logprob < %.3f", *cascadeMinLP)
			}
			return c
		}
		log.Fatalf("unknown architecture: %s (expected closedbook | naive | ircot | crag | rerank | hyde | bm25 | hybrid | fused | adaptive | neighbour | cascade)", arch)
		return nil
	}
	pipeline := build(*architecture)
	log.Printf("architecture: %s (top-k %d, %d candidates)", *architecture, *topK, *candidates)

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
				results[i] = evaluate(ctx, pipeline, dataset[i], *abstentionWords)
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

	sum := aggregate(results)
	if sum.answered == 0 {
		log.Fatalf("every question failed (%d errors) - check that the collection %q exists and both model servers are up", failed, *collection)
	}

	if *dumpPath != "" {
		if err := writeDump(*dumpPath, results); err != nil {
			log.Fatalf("writing the dump file: %v", err)
		}
		log.Printf("per-question detail written to %s", *dumpPath)
	}

	ci := func(values []float64) string {
		if *resamples <= 0 || len(values) == 0 {
			return ""
		}
		lo, hi := metrics.BootstrapCI(values, *resamples, rand.New(rand.NewSource(*seed)))
		return fmt.Sprintf("  95%% CI [%.4f, %.4f]", lo, hi)
	}

	fmt.Println("--- results ---")
	fmt.Printf("dataset:            %s\n", *datasetPath)
	fmt.Printf("collection:         %s (top-k=%d, hnsw_ef=%d)\n", *collection, *topK, *hnswEf)
	fmt.Printf("architecture:       %s\n", *architecture)
	fmt.Printf("embedding model:    %s (query prefix %q)\n", *embedModel, resolvedPrefix)
	fmt.Printf("llm:                %s (temperature=%g, answer/step token caps=%d/%d)\n", *llmModel, *temperature, *maxAnswer, *maxStep)
	fmt.Printf("questions:          %d evaluated", sum.answered)
	if failed > 0 {
		fmt.Printf(", %d failed", failed)
	}
	fmt.Println()
	fmt.Printf("Exact Match:        %.4f%s\n", sum.em, ci(sum.emVals))
	fmt.Printf("F1:                 %.4f%s\n", sum.f1, ci(sum.f1Vals))
	fmt.Printf("Abstention rate:    %.4f%s (declined instead of answering)\n", sum.abstention, ci(sum.abstainVals))
	if sum.answeredOnly > 0 {
		fmt.Printf("EM (answered only): %.4f  over the %d/%d questions the system did not decline\n", sum.emAnswered, sum.answeredOnly, sum.answered)
		fmt.Printf("F1 (answered only): %.4f\n", sum.f1Answered)
	}
	if sum.meanPassages > 0 {
		fmt.Printf("Passages in ctx:    %.2f on average\n", sum.meanPassages)
		if sum.inCtxN > 0 {
			fmt.Printf("Answer in context:  %.4f%s (over %d/%d questions where a gold answer is a span in the text)\n", sum.inCtx, ci(sum.inCtxVals), sum.inCtxN, sum.answered)
		}
		if sum.goldN > 0 {
			fmt.Printf("Recall (context):   %.4f%s (gold articles among the %.2f passages in context, over %d/%d annotated questions)\n", sum.recall, ci(sum.recallVals), sum.meanPassages, sum.goldN, sum.answered)
			fmt.Printf("MRR:                %.4f%s\n", sum.mrr, ci(sum.rrVals))
		} else {
			fmt.Printf("Recall/MRR:         not computed (this dataset annotates no gold documents)\n")
		}
	} else {
		fmt.Printf("retrieval metrics:  not computed (this architecture retrieves nothing)\n")
	}
	fmt.Printf("LLM calls:          %.2f per question\n", sum.llmCalls)
	fmt.Printf("Tokens:             %.0f prompt + %.0f completion per question\n", sum.promptTokens, sum.completionTokens)
	fmt.Printf("wall clock:         %s (%.1f questions/s, concurrency=%d)\n", wall.Round(time.Second), float64(sum.answered)/wall.Seconds(), *concurrency)
	if *concurrency == 1 {
		fmt.Printf("Avg latency:        %s\n", sum.latency.Round(time.Millisecond))
	} else {
		fmt.Printf("Avg latency:        %s - NOT a latency measurement at concurrency=%d, it includes queueing. Re-run with -concurrency 1 on a subsample to report latency.\n", sum.latency.Round(time.Millisecond), *concurrency)
	}
	if len(sum.routes) > 0 {
		fmt.Printf("routes:            ")
		printDistribution(sum.routes)
	}
	if len(sum.branches) > 0 {
		fmt.Printf("crag branches:     ")
		printDistribution(sum.branches)
	}
	if len(sum.stages) > 0 {
		fmt.Printf("answered by stage: ")
		printDistribution(sum.stages)
		fmt.Printf("stages run:         %.2f per question\n", sum.meanStagesRun)
	}

	if *texOut != "" {
		sum.name = *name
		sum.topK = *topK
		sum.hnswEf = *hnswEf
		sum.concurrency = *concurrency
		sum.throughput = float64(sum.answered) / wall.Seconds()
		if err := writeTex(*texOut, sum); err != nil {
			log.Fatalf("writing the tex file: %v", err)
		}
		log.Printf("results written to %s", *texOut)
	}
}

func printDistribution(counts map[string]int64) {
	total := 0.0
	for _, c := range counts {
		total += float64(c)
	}
	for _, key := range slices.Sorted(maps.Keys(counts)) {
		fmt.Printf(" %s=%d (%.1f%%)", key, counts[key], 100*float64(counts[key])/total)
	}
	fmt.Println()
}

// evaluate runs one question through the pipeline and scores it.
//
// Retrieval is scored at the article level: a Wikipedia article is split into
// many ~100-word passages sharing one title, and the datasets annotate gold
// *articles*, not passages. Counting passages instead would cap Recall@5 at
// roughly 0.31 on HotpotQA - measured - producing a number that cannot be
// compared with any published Recall@5.
func evaluate(ctx context.Context, pipeline Pipeline, item datasetLine, abstentionWords int) result {
	// The usage and trace accumulators ride on the context, so an
	// architecture that makes several generation calls through code that
	// knows nothing about measurement still reports its true cost.
	ctx, usage := provider.WithUsage(ctx)
	ctx, trace := rag.WithTrace(ctx)
	ctx, confidence := provider.WithConfidence(ctx)

	start := time.Now()
	answer, passages, err := pipeline.Query(ctx, item.Question)
	latency := time.Since(start)
	if err != nil {
		log.Printf("question %q: query failed: %v", item.Question, err)
		return result{}
	}

	texts := make([]string, len(passages))
	ids := make([]uint64, len(passages))
	titles := make([]string, len(passages))
	for i, p := range passages {
		texts[i] = p.Text
		ids[i] = p.ID
		titles[i] = flashrag.PassageTitle(p.Text)
	}

	r := result{
		ok:               true,
		question:         item.Question,
		answer:           answer,
		gold:             item.GoldenAnswers,
		passages:         len(passages),
		retrievedIDs:     ids,
		retrievedTitles:  titles,
		em:               metrics.ExactMatch(answer, item.GoldenAnswers),
		f1:               metrics.F1Score(answer, item.GoldenAnswers),
		abstained:        metrics.IsAbstention(answer, abstentionWords),
		latency:          latency,
		promptTokens:     usage.PromptTokens.Load(),
		completionTokens: usage.CompletionTokens.Load(),
		llmCalls:         usage.Calls.Load(),
		trace:            trace.Snapshot(),
	}
	if mean, ok := confidence.Last(); ok {
		r.meanLogprob = &mean
	}

	if metrics.AnswerInContextApplicable(item.GoldenAnswers) {
		r.inCtxOK = true
		r.inCtx = metrics.AnswerInContext(texts, item.GoldenAnswers)
	}

	if gold := flashrag.GoldTitles(item.Metadata); len(gold) > 0 {
		r.hasGold = true
		// Titles are normalised on both sides: the corpus and the question
		// sets disagree on Unicode composition often enough that a byte
		// comparison silently loses 4.5% of 2WikiMultihopQA's gold articles.
		//
		// Scored over every passage placed in the generator's context, not
		// over the first topK. An architecture that accumulates passages
		// across several retrieval rounds (IRCoT) puts more than topK in
		// context; trimming to topK would discard exactly the passages its
		// extra rounds fetched and report it as identical to a single-query
		// baseline. This is why the reported figure is Recall over the
		// context, with the mean context size alongside it, and why the
		// matched-budget controls (naive10, naive12) exist.
		normalisedTitles := flashrag.NormalizeTitles(titles)
		normalisedGold := flashrag.NormalizeTitles(gold)
		r.recall = metrics.RecallAtK(normalisedTitles, normalisedGold, len(normalisedTitles))
		r.rr = metrics.ReciprocalRank(normalisedTitles, normalisedGold)
	}
	return r
}

// summary is everything the run reports, so that adding a measurement does not
// mean growing an eleven-argument writeTex.
type summary struct {
	name        string
	topK        int
	hnswEf      int
	concurrency int
	throughput  float64

	answered     int
	answeredOnly int
	inCtxN       int
	goldN        int

	em, f1                 float64
	emAnswered, f1Answered float64
	abstention             float64
	inCtx                  float64
	recall, mrr            float64
	meanPassages           float64
	promptTokens           float64
	completionTokens       float64
	llmCalls               float64
	latency                time.Duration

	routes        map[string]int64
	branches      map[string]int64
	stages        map[string]int64
	meanStagesRun float64

	emVals, f1Vals, abstainVals, inCtxVals, recallVals, rrVals []float64
}

// aggregate reduces the per-question results.
//
// Only answered questions contribute; a failed query is excluded rather than
// scored as zero, so an outage cannot masquerade as a bad result. Each metric
// is averaged over the questions it applies to rather than over all of them,
// which is why the counts are reported next to the values.
func aggregate(results []result) summary {
	s := summary{routes: map[string]int64{}, branches: map[string]int64{}, stages: map[string]int64{}}
	var sumStagesRun float64

	var sumEM, sumF1, sumAbstain, sumInCtx, sumRecall, sumRR, sumPassages float64
	var sumEMAnswered, sumF1Answered float64
	var sumPrompt, sumCompletion, sumCalls float64
	var sumLatency time.Duration

	for _, r := range results {
		if !r.ok {
			continue
		}
		s.answered++
		sumEM += r.em
		sumF1 += r.f1
		sumPassages += float64(r.passages)
		sumLatency += r.latency
		sumPrompt += float64(r.promptTokens)
		sumCompletion += float64(r.completionTokens)
		sumCalls += float64(r.llmCalls)
		s.emVals = append(s.emVals, r.em)
		s.f1Vals = append(s.f1Vals, r.f1)

		abstained := 0.0
		if r.abstained {
			abstained = 1
		} else {
			s.answeredOnly++
			sumEMAnswered += r.em
			sumF1Answered += r.f1
		}
		sumAbstain += abstained
		s.abstainVals = append(s.abstainVals, abstained)

		if r.inCtxOK {
			s.inCtxN++
			sumInCtx += r.inCtx
			s.inCtxVals = append(s.inCtxVals, r.inCtx)
		}
		if r.hasGold {
			s.goldN++
			sumRecall += r.recall
			sumRR += r.rr
			s.recallVals = append(s.recallVals, r.recall)
			s.rrVals = append(s.rrVals, r.rr)
		}
		if r.trace.Route != "" {
			s.routes[r.trace.Route]++
		}
		if r.trace.Branch != "" {
			s.branches[r.trace.Branch]++
		}
		if r.trace.Stage != "" {
			s.stages[r.trace.Stage]++
			sumStagesRun += float64(r.trace.StagesRun)
		}
	}
	if s.answered == 0 {
		return s
	}

	n := float64(s.answered)
	s.em = sumEM / n
	s.f1 = sumF1 / n
	s.abstention = sumAbstain / n
	s.meanPassages = sumPassages / n
	s.promptTokens = sumPrompt / n
	s.completionTokens = sumCompletion / n
	s.llmCalls = sumCalls / n
	s.latency = time.Duration(float64(sumLatency) / n)
	if s.answeredOnly > 0 {
		s.emAnswered = sumEMAnswered / float64(s.answeredOnly)
		s.f1Answered = sumF1Answered / float64(s.answeredOnly)
	}
	if s.inCtxN > 0 {
		s.inCtx = sumInCtx / float64(s.inCtxN)
	}
	if s.goldN > 0 {
		s.recall = sumRecall / float64(s.goldN)
		s.mrr = sumRR / float64(s.goldN)
	}
	if len(s.stages) > 0 {
		s.meanStagesRun = sumStagesRun / n
	}
	return s
}

// writeTex writes the results as a LaTeX fragment (\newcommand definitions),
// ready to \input{} from the body of the paper, e.g.:
//
//	\input{baseline.gen.tex}
//	Exact Match: \NaiveRAGEM, F1: \NaiveRAGFOne
//
// A measurement that was not made is not emitted as zero. ClosedBook retrieves
// nothing, so a \ClosedBookRecall of 0.0000 in a table reads as a measured
// failure when it is a definition; those commands are simply absent, and a
// table that references one fails to compile instead of printing a fiction.
func writeTex(path string, s summary) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}

	id := texSafeID(s.name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/bench - do not edit by hand\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sQuestions}{%d}\n", id, s.answered)
	fmt.Fprintf(&b, "\\newcommand{\\%sEM}{%.4f}\n", id, s.em)
	fmt.Fprintf(&b, "\\newcommand{\\%sFOne}{%.4f}\n", id, s.f1)
	fmt.Fprintf(&b, "\\newcommand{\\%sAbstentionRate}{%.4f}\n", id, s.abstention)
	if s.answeredOnly > 0 {
		fmt.Fprintf(&b, "\\newcommand{\\%sAnsweredQuestions}{%d}\n", id, s.answeredOnly)
		fmt.Fprintf(&b, "\\newcommand{\\%sEMAnswered}{%.4f}\n", id, s.emAnswered)
		fmt.Fprintf(&b, "\\newcommand{\\%sFOneAnswered}{%.4f}\n", id, s.f1Answered)
	}

	if s.meanPassages > 0 {
		fmt.Fprintf(&b, "\\newcommand{\\%sMeanPassagesInContext}{%.2f}\n", id, s.meanPassages)
		fmt.Fprintf(&b, "\\newcommand{\\%sTopK}{%d}\n", id, s.topK)
		fmt.Fprintf(&b, "\\newcommand{\\%sHNSWEf}{%d}\n", id, s.hnswEf)
		if s.inCtxN > 0 {
			fmt.Fprintf(&b, "\\newcommand{\\%sAnswerInContext}{%.4f}\n", id, s.inCtx)
			fmt.Fprintf(&b, "\\newcommand{\\%sAnswerInContextQuestions}{%d}\n", id, s.inCtxN)
		}
		if s.goldN > 0 {
			// Named for what it measures: recall over the passages that were
			// in context, whose mean size is reported next to it. Calling it
			// Recall@K next to a TopK of 5 would claim a same-K comparison
			// that a multi-round architecture does not make.
			fmt.Fprintf(&b, "\\newcommand{\\%sRecallInContext}{%.4f}\n", id, s.recall)
			fmt.Fprintf(&b, "\\newcommand{\\%sMRR}{%.4f}\n", id, s.mrr)
			fmt.Fprintf(&b, "\\newcommand{\\%sRetrievalEvalQuestions}{%d}\n", id, s.goldN)
		}
	}

	fmt.Fprintf(&b, "\\newcommand{\\%sLLMCalls}{%.2f}\n", id, s.llmCalls)
	fmt.Fprintf(&b, "\\newcommand{\\%sPromptTokens}{%.0f}\n", id, s.promptTokens)
	fmt.Fprintf(&b, "\\newcommand{\\%sCompletionTokens}{%.0f}\n", id, s.completionTokens)
	fmt.Fprintf(&b, "\\newcommand{\\%sThroughput}{%.2f}\n", id, s.throughput)
	fmt.Fprintf(&b, "\\newcommand{\\%sConcurrency}{%d}\n", id, s.concurrency)
	// Latency is only a latency at concurrency 1. Above it the per-question
	// timing includes queueing behind the other in-flight questions, so the
	// command is not emitted and a table cannot quote it by accident.
	if s.concurrency == 1 {
		fmt.Fprintf(&b, "\\newcommand{\\%sLatency}{%s}\n", id, s.latency.Round(time.Millisecond))
	}

	for _, key := range slices.Sorted(maps.Keys(s.routes)) {
		fmt.Fprintf(&b, "\\newcommand{\\%sRoute%s}{%d}\n", id, texSafeID(capitalise(key)), s.routes[key])
	}
	for _, key := range slices.Sorted(maps.Keys(s.branches)) {
		fmt.Fprintf(&b, "\\newcommand{\\%sBranch%s}{%d}\n", id, texSafeID(capitalise(key)), s.branches[key])
	}
	for _, key := range slices.Sorted(maps.Keys(s.stages)) {
		fmt.Fprintf(&b, "\\newcommand{\\%sStage%s}{%d}\n", id, texSafeID(capitalise(key)), s.stages[key])
	}
	if len(s.stages) > 0 {
		fmt.Fprintf(&b, "\\newcommand{\\%sStagesRun}{%.2f}\n", id, s.meanStagesRun)
	}

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func capitalise(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
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

// writeDump records the per-question detail as JSONL, one object per line, in
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

	w := bufio.NewWriterSize(f, 1024*1024)
	enc := json.NewEncoder(w)
	for i, r := range results {
		if !r.ok {
			continue
		}
		if err := enc.Encode(dumpRecord{
			Index:            i,
			Question:         r.question,
			Answer:           r.answer,
			Gold:             r.gold,
			EM:               r.em,
			F1:               r.f1,
			Abstained:        r.abstained,
			InCtx:            r.inCtx,
			InCtxApplicable:  r.inCtxOK,
			HasGold:          r.hasGold,
			Recall:           r.recall,
			RR:               r.rr,
			Passages:         r.passages,
			RetrievedIDs:     r.retrievedIDs,
			RetrievedTitles:  r.retrievedTitles,
			PromptTokens:     r.promptTokens,
			CompletionTokens: r.completionTokens,
			LLMCalls:         r.llmCalls,
			Route:            r.trace.Route,
			Branch:           r.trace.Branch,
			GradedKept:       r.trace.GradedKept,
			RewrittenQuery:   r.trace.RewrittenQuery,
			ReasoningSteps:   r.trace.Steps,
			Stage:            r.trace.Stage,
			StagesRun:        r.trace.StagesRun,
			Escalations:      r.trace.Escalations,
			MeanLogprob:      r.meanLogprob,
			LatencyS:         r.latency.Seconds(),
		}); err != nil {
			return err
		}
	}
	return w.Flush()
}
