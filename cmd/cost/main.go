// Command cost renders the cost table: what each architecture spends per
// question, read from the per-question dumps rather than from timings.
//
// Generation calls and tokens are the only cost measure that is comparable
// across hardware. A latency measured at -concurrency 48 contains queueing,
// and the runs here were made at different times against a shared card, so
// two runs with identical token counts differ in throughput by a factor of
// two. Tokens do not. The prompt tokens are the figure that matters: a
// passage is ~170 tokens and an answer a handful, so the prompt is where the
// money goes, and it is reported relative to the standard baseline so that a
// reader sees "1.15x NaiveRAG" without doing the division.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/report"
)

type record struct {
	LLMCalls         int64 `json:"llm_calls"`
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	Passages         int   `json:"passages_in_context"`
}

// row is one architecture, averaged over the sets it ran on; per-set means
// are kept for the range of calls, which varies with how often a cascade
// escalates.
type row struct {
	slug                    string
	sets                    int
	calls, prompt, complete float64
	passages                float64
	callsLo, callsHi        float64
}

func main() {
	var (
		runsDir  = flag.String("runs", "runs", "directory holding the -dump files, named <architecture>-<set>.jsonl")
		archs    = flag.String("architectures", "closedbook,bm25,naive,naive10,naive12,hybrid,hyde,rerank,neighbour,crag,crag10,ircot,adaptive,fused,cascade", "comma-separated slugs, in table order")
		sets     = flag.String("sets", "nq,triviaqa,hotpotqa,2wiki,musique", "comma-separated question-set slugs")
		relative = flag.String("relative", "naive", "slug of the architecture the prompt-token ratio is taken against")
		suffix   = flag.String("suffix", "", "tag inserted after every slug in the file names, e.g. \"llama\" reads naive-llama-<set>.jsonl")
		texOut   = flag.String("tex-out", "", "path of a .tex file to write the table to - optional")
		name     = flag.String("name", "Cost", "prefix of the generated .tex commands")
	)
	flag.Parse()

	tag := ""
	if *suffix != "" {
		tag = "-" + *suffix
	}
	var rows []row
	for _, a := range strings.Split(*archs, ",") {
		r := row{slug: strings.TrimSpace(a), callsLo: math.Inf(1), callsHi: math.Inf(-1)}
		for _, s := range strings.Split(*sets, ",") {
			recs, ok := load(filepath.Join(*runsDir, r.slug+tag+"-"+strings.TrimSpace(s)+".jsonl"))
			if !ok || len(recs) == 0 {
				continue
			}
			var calls, prompt, complete, passages float64
			for _, x := range recs {
				calls += float64(x.LLMCalls)
				prompt += float64(x.PromptTokens)
				complete += float64(x.CompletionTokens)
				passages += float64(x.Passages)
			}
			n := float64(len(recs))
			r.sets++
			r.calls += calls / n
			r.prompt += prompt / n
			r.complete += complete / n
			r.passages += passages / n
			r.callsLo = min(r.callsLo, calls/n)
			r.callsHi = max(r.callsHi, calls/n)
		}
		if r.sets == 0 {
			continue
		}
		n := float64(r.sets)
		r.calls, r.prompt, r.complete, r.passages = r.calls/n, r.prompt/n, r.complete/n, r.passages/n
		rows = append(rows, r)
	}
	if len(rows) == 0 {
		log.Fatalf("no dumps found under %s", *runsDir)
	}

	base := math.NaN()
	for _, r := range rows {
		if r.slug == *relative {
			base = r.prompt
		}
	}

	fmt.Printf("%-34s %5s %10s %8s %8s %9s %6s\n", "architecture", "sets", "calls", "prompt", "compl.", "passages", "x"+*relative)
	for _, r := range rows {
		fmt.Printf("%-34s %5d %10s %8.0f %8.0f %9.2f %6s\n", report.ArchitectureName(r.slug), r.sets, callsLabel(r), r.prompt, r.complete, r.passages, ratio(r.prompt, base))
	}
	fmt.Printf("\nmeans over the sets each architecture ran on; calls shown as a range when they differ across sets.\n")

	if *texOut != "" {
		if err := writeTex(*texOut, *name, *relative, rows, base); err != nil {
			log.Fatalf("writing tex: %v", err)
		}
		log.Printf("written to %s", *texOut)
	}
}

func callsLabel(r row) string {
	if r.callsHi-r.callsLo < 0.05 {
		return fmt.Sprintf("%.2f", r.calls)
	}
	return fmt.Sprintf("%.2f-%.2f", r.callsLo, r.callsHi)
}

func ratio(v, base float64) string {
	if math.IsNaN(base) || base == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", v/base)
}

func writeTex(path, name, relative string, rows []row, base float64) error {
	id := texSafeID(name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/cost - do not edit by hand\n")
	fmt.Fprintf(&b, "%% Per-question means over the sets each architecture ran on. Calls are a range when they differ across sets.\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sRelativeTo}{%s}\n", id, report.ArchitectureName(relative))
	for _, r := range rows {
		rid := texSafeID(r.slug)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sCalls}{%.2f}\n", id, rid, r.calls)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sPromptTokens}{%.0f}\n", id, rid, r.prompt)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sCompletionTokens}{%.0f}\n", id, rid, r.complete)
		if !math.IsNaN(base) && base != 0 {
			fmt.Fprintf(&b, "\\newcommand{\\%s%sPromptRatio}{%.2f}\n", id, rid, r.prompt/base)
		}
	}
	fmt.Fprintf(&b, "\\newcommand{\\%sTable}{%%\n", id)
	fmt.Fprintf(&b, "\\begin{tabular}{lrrrrr}\n\\toprule\n")
	fmt.Fprintf(&b, "Architecture & LLM calls & Prompt tokens & Completion tokens & Passages & $\\times$ %s \\\\\n\\midrule\n", report.ArchitectureName(relative))
	for _, r := range rows {
		fmt.Fprintf(&b, "%s & %s & %.0f & %.0f & %.1f & %s \\\\\n", report.ArchitectureName(r.slug), callsLabel(r), r.prompt, r.complete, r.passages, ratio(r.prompt, base))
	}
	fmt.Fprintf(&b, "\\bottomrule\n\\end{tabular}}\n")
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// texSafeID keeps letters only, mapping the digits a slug may carry to words
// so that naive10 and naive12 stay distinct command names.
func texSafeID(name string) string {
	digits := map[rune]string{'0': "Zero", '1': "One", '2': "Two", '3': "Three", '4': "Four", '5': "Five", '6': "Six", '7': "Seven", '8': "Eight", '9': "Nine"}
	var b strings.Builder
	for _, r := range name {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteString(digits[r])
		}
	}
	return b.String()
}

func load(path string) ([]record, bool) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false
		}
		log.Fatalf("reading %s: %v", path, err)
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
			log.Fatalf("%s: parse line: %v", path, err)
		}
		out = append(out, r)
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("reading %s: %v", path, err)
	}
	return out, true
}
