// Command aggregate recomputes a benchmark run's eval document from its
// per-question dump, without touching the GPU.
//
// cmd/bench writes the same aggregates at the end of a live run; this
// command exists for the runs that already happened. Renaming an aggregate,
// changing a format, or adding a derived figure would otherwise mean
// re-running hours of benchmarks whose per-question outcomes have not
// changed at all - everything the paper cites is already in the dump.
//
// It mirrors cmd/bench's aggregation over the dump schema, with one
// difference: figures that describe the run rather than the questions
// (throughput, concurrency, top-k, HNSW ef, latency) are not in the dump and
// are not emitted. The paper cites none of them; a live cmd/bench run is the
// only source that records them.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"maps"
	"os"
	"slices"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/eval"
)

// record is one line of a cmd/bench -dump file, reduced to the fields the
// aggregates are computed from.
type record struct {
	EM        float64 `json:"em"`
	F1        float64 `json:"f1"`
	Abstained bool    `json:"abstained"`

	InCtx           float64 `json:"answer_in_context"`
	InCtxApplicable bool    `json:"answer_in_context_applicable"`
	HasGold         bool    `json:"has_gold"`
	Recall          float64 `json:"recall_in_context"`
	RR              float64 `json:"reciprocal_rank"`

	Passages         int   `json:"passages_in_context"`
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	LLMCalls         int64 `json:"llm_calls"`

	Route     string `json:"route"`
	Branch    string `json:"branch"`
	Stage     string `json:"stage"`
	StagesRun int    `json:"stages_run"`
}

func main() {
	var (
		dumpPath = flag.String("dump", "", "cmd/bench -dump file to aggregate (required)")
		name     = flag.String("name", "", "run name prefixing the aggregates, exactly as the run's cmd/bench -name (required)")
		jsonOut  = flag.String("json-out", "", "path of the eval .json file to write (required)")
	)
	flag.Parse()
	if *dumpPath == "" || *name == "" || *jsonOut == "" {
		log.Fatal("-dump, -name and -json-out are all required")
	}

	records, err := load(*dumpPath)
	if err != nil {
		log.Fatalf("reading %s: %v", *dumpPath, err)
	}
	if len(records) == 0 {
		log.Fatalf("%s holds no records", *dumpPath)
	}

	if err := eval.Write(*jsonOut, aggregate(*name, records)); err != nil {
		log.Fatalf("writing the eval file: %v", err)
	}
	log.Printf("aggregated %d questions from %s into %s (run cmd/render to refresh the LaTeX fragments)", len(records), *dumpPath, *jsonOut)
}

// aggregate mirrors cmd/bench: every metric is averaged over the questions
// it applies to, and a measurement that was not made is not emitted as zero.
func aggregate(name string, records []record) *eval.Doc {
	var sumEM, sumF1, sumAbstain, sumInCtx, sumRecall, sumRR, sumPassages float64
	var sumEMAnswered, sumF1Answered float64
	var sumPrompt, sumCompletion, sumCalls, sumStagesRun float64
	var answeredOnly, inCtxN, goldN int
	routes, branches, stages := map[string]int64{}, map[string]int64{}, map[string]int64{}

	for _, r := range records {
		sumEM += r.EM
		sumF1 += r.F1
		sumPassages += float64(r.Passages)
		sumPrompt += float64(r.PromptTokens)
		sumCompletion += float64(r.CompletionTokens)
		sumCalls += float64(r.LLMCalls)
		if r.Abstained {
			sumAbstain++
		} else {
			answeredOnly++
			sumEMAnswered += r.EM
			sumF1Answered += r.F1
		}
		if r.InCtxApplicable {
			inCtxN++
			sumInCtx += r.InCtx
		}
		if r.HasGold {
			goldN++
			sumRecall += r.Recall
			sumRR += r.RR
		}
		if r.Route != "" {
			routes[r.Route]++
		}
		if r.Branch != "" {
			branches[r.Branch]++
		}
		if r.Stage != "" {
			stages[r.Stage]++
			sumStagesRun += float64(r.StagesRun)
		}
	}

	n := float64(len(records))
	id := texSafeID(name)
	doc := &eval.Doc{Generator: "cmd/aggregate"}
	doc.Addf(id+"Questions", "%d", len(records))
	doc.Addf(id+"EM", "%.4f", sumEM/n)
	doc.Addf(id+"FOne", "%.4f", sumF1/n)
	doc.Addf(id+"AbstentionRate", "%.4f", sumAbstain/n)
	if answeredOnly > 0 {
		doc.Addf(id+"AnsweredQuestions", "%d", answeredOnly)
		doc.Addf(id+"EMAnswered", "%.4f", sumEMAnswered/float64(answeredOnly))
		doc.Addf(id+"FOneAnswered", "%.4f", sumF1Answered/float64(answeredOnly))
	}
	if sumPassages > 0 {
		doc.Addf(id+"MeanPassagesInContext", "%.2f", sumPassages/n)
		if inCtxN > 0 {
			doc.Addf(id+"AnswerInContext", "%.4f", sumInCtx/float64(inCtxN))
			doc.Addf(id+"AnswerInContextQuestions", "%d", inCtxN)
		}
		if goldN > 0 {
			doc.Addf(id+"RecallInContext", "%.4f", sumRecall/float64(goldN))
			doc.Addf(id+"MRR", "%.4f", sumRR/float64(goldN))
			doc.Addf(id+"RetrievalEvalQuestions", "%d", goldN)
		}
	}
	doc.Addf(id+"LLMCalls", "%.2f", sumCalls/n)
	doc.Addf(id+"PromptTokens", "%.0f", sumPrompt/n)
	doc.Addf(id+"CompletionTokens", "%.0f", sumCompletion/n)

	for _, key := range slices.Sorted(maps.Keys(routes)) {
		doc.Addf(id+"Route"+texSafeID(capitalise(key)), "%d", routes[key])
	}
	for _, key := range slices.Sorted(maps.Keys(branches)) {
		doc.Addf(id+"Branch"+texSafeID(capitalise(key)), "%d", branches[key])
	}
	for _, key := range slices.Sorted(maps.Keys(stages)) {
		doc.Addf(id+"Stage"+texSafeID(capitalise(key)), "%d", stages[key])
	}
	if len(stages) > 0 {
		doc.Addf(id+"StagesRun", "%.2f", sumStagesRun/n)
	}
	return doc
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

func load(path string) ([]record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []record
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, fmt.Errorf("parse line: %w", err)
		}
		out = append(out, r)
	}
	return out, scanner.Err()
}
