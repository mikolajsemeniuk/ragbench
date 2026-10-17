// Command sweep chooses the cascade's confidence threshold without touching
// the GPU or the test sets.
//
// The cascade can escalate an answer for two reasons: the reader declined
// (lossless), or the reader's mean token log-probability fell below a
// threshold (not lossless - it can discard a correct answer). The threshold is
// a hyperparameter, and a hyperparameter chosen by looking at the test-set
// score is the first thing a reviewer strikes. So it is chosen here, on dumps
// of the TRAIN splits, by composing the stages question by question: keep the
// first stage's answer unless it abstained or its confidence is below the
// threshold, otherwise the second stage's under the same rule, otherwise the
// last stage's. Composition is exact under greedy decoding because every stage
// is deterministic given the question.
//
// One value is chosen for all question sets, because a per-set threshold would
// need to know which set a question came from. Each of -stage1, -stage2 and
// -last therefore accepts a comma-separated list of dumps, one per set, and
// the sweep is over the pooled questions.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/eval"
)

type record struct {
	Index       int      `json:"index"`
	Question    string   `json:"question"`
	EM          float64  `json:"em"`
	F1          float64  `json:"f1"`
	Abstained   bool     `json:"abstained"`
	LLMCalls    int64    `json:"llm_calls"`
	MeanLogprob *float64 `json:"mean_logprob"`
}

type point struct {
	threshold                    float64
	em, f1, calls                float64
	escalated, discarded, missed int
}

func main() {
	var (
		stage1     = flag.String("stage1", "", "dump(s) of the first stage run on its own, with mean_logprob; comma-separated, one per set (required)")
		stage2     = flag.String("stage2", "", "dump(s) of the second stage on the same questions, in the same order - optional")
		last       = flag.String("last", "", "dump(s) of the last stage on the same questions, in the same order (required)")
		thresholds = flag.String("thresholds", "-2,-1.5,-1,-0.8,-0.6,-0.5,-0.4,-0.3,-0.2,-0.15,-0.1,-0.05,-0.02,-0.01,-0.005", "comma-separated mean log-probability thresholds to try")
		jsonOut    = flag.String("json-out", "", "path of the eval .json file to write the sweep to (rendered to LaTeX by cmd/render) - optional")
		name       = flag.String("name", "Sweep", "prefix of the aggregate names")
	)
	flag.Parse()
	if *stage1 == "" || *last == "" {
		log.Fatal("-stage1 and -last are required")
	}

	s1 := loadAll(*stage1)
	lastRun := loadAll(*last)
	var s2 map[key]record
	if *stage2 != "" {
		s2 = loadAll(*stage2)
	}

	var keys []key
	for k := range s1 {
		if _, ok := lastRun[k]; !ok {
			continue
		}
		if s2 != nil {
			if _, ok := s2[k]; !ok {
				continue
			}
		}
		keys = append(keys, k)
	}
	if len(keys) == 0 {
		log.Fatal("the dumps share no questions - were they made with the same -limit and -seed?")
	}
	withConf := 0
	for _, k := range keys {
		if s1[k].MeanLogprob != nil {
			withConf++
		}
	}
	if withConf == 0 {
		log.Fatalf("%s has no mean_logprob field - it predates the confidence signal and has to be regenerated", *stage1)
	}

	values := []float64{math.Inf(-1)}
	for _, f := range strings.Split(*thresholds, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
		if err != nil {
			log.Fatalf("bad threshold %q: %v", f, err)
		}
		values = append(values, v)
	}

	var points []point
	for _, t := range values {
		points = append(points, simulate(t, keys, s1, s2, lastRun))
	}

	fmt.Printf("questions: %d (%d with a confidence value)\n\n", len(keys), withConf)
	fmt.Printf("%-10s %8s %8s %8s %10s %10s %8s\n", "threshold", "EM", "F1", "calls", "escalated", "discarded", "missed")
	best := 0
	for i, p := range points {
		label := "abst-only"
		if !math.IsInf(p.threshold, -1) {
			label = fmt.Sprintf("%.3f", p.threshold)
		}
		fmt.Printf("%-10s %8.4f %8.4f %8.2f %10d %10d %8d\n", label, p.em, p.f1, p.calls, p.escalated, p.discarded, p.missed)
		if p.em > points[best].em {
			best = i
		}
	}
	fmt.Printf("\n'escalated' counts stage-1 answers handed on, 'discarded' the correct ones among them, 'missed' those a later stage did not get back.\n")
	if best == 0 {
		fmt.Printf("best Exact Match is the abstention-only cascade: the confidence signal does not pay on these questions.\n")
	} else {
		fmt.Printf("best Exact Match at threshold %.2f: %+.4f over abstention-only for %+.2f calls per question.\n", points[best].threshold, points[best].em-points[0].em, points[best].calls-points[0].calls)
	}

	if *jsonOut != "" {
		if err := writeJSON(*jsonOut, *name, len(keys), points, best); err != nil {
			log.Fatalf("writing eval file: %v", err)
		}
		log.Printf("written to %s", *jsonOut)
	}
}

// simulate composes the stages at one threshold.
func simulate(threshold float64, keys []key, s1, s2, last map[key]record) point {
	p := point{threshold: threshold}
	n := float64(len(keys))
	escalate := func(r record) bool {
		if r.Abstained {
			return true
		}
		return r.MeanLogprob != nil && *r.MeanLogprob < threshold
	}
	for _, k := range keys {
		first := s1[k]
		calls := float64(first.LLMCalls)
		chosen := first
		if escalate(first) {
			p.escalated++
			if first.EM >= 1 {
				p.discarded++
			}
			chosen = last[k]
			calls += float64(last[k].LLMCalls)
			if s2 != nil {
				second := s2[k]
				calls = float64(first.LLMCalls) + float64(second.LLMCalls)
				if escalate(second) {
					calls += float64(last[k].LLMCalls)
				} else {
					chosen = second
				}
			}
			if first.EM >= 1 && chosen.EM < 1 {
				p.missed++
			}
		}
		p.em += chosen.EM
		p.f1 += chosen.F1
		p.calls += calls
	}
	p.em /= n
	p.f1 /= n
	p.calls /= n
	return p
}

func writeJSON(path, name string, questions int, points []point, best int) error {
	id := texSafeID(name)
	doc := eval.Doc{Generator: "cmd/sweep"}
	doc.Addf(id+"Questions", "%d", questions)
	bestLabel := "$-\\infty$"
	if !math.IsInf(points[best].threshold, -1) {
		bestLabel = fmt.Sprintf("%.3f", points[best].threshold)
	}
	doc.Add(id+"BestThreshold", bestLabel)
	doc.Addf(id+"BestEM", "%.4f", points[best].em)
	doc.Addf(id+"AbstentionOnlyEM", "%.4f", points[0].em)

	var b strings.Builder
	fmt.Fprintf(&b, "%%\n")
	for _, p := range points {
		label := "$-\\infty$ (abstention only)"
		if !math.IsInf(p.threshold, -1) {
			label = fmt.Sprintf("%.3f", p.threshold)
		}
		fmt.Fprintf(&b, "%s & %.4f & %.4f & %.2f & %d & %d & %d \\\\\n", label, p.em, p.f1, p.calls, p.escalated, p.discarded, p.missed)
	}
	doc.Add(id+"Rows", b.String())

	return eval.Write(path, &doc)
}

func texSafeID(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// key identifies a question across dumps. The file ordinal is part of it
// because indices restart at 0 in every question set, and the sweep pools
// several sets.
type key struct {
	file     int
	index    int
	question string
}

// loadAll reads a comma-separated list of dumps into one map.
func loadAll(paths string) map[key]record {
	out := map[key]record{}
	for i, path := range strings.Split(paths, ",") {
		for k, r := range load(strings.TrimSpace(path)) {
			k.file = i
			out[k] = r
		}
	}
	return out
}

func load(path string) map[key]record {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("reading %s: %v", path, err)
	}
	defer f.Close()
	out := map[key]record{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			log.Fatalf("%s: parse line: %v", path, err)
		}
		out[key{index: r.Index, question: r.Question}] = r
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("reading %s: %v", path, err)
	}
	return out
}
