// Command result renders the paper's summary table: the proposed architecture
// against every baseline on every question set, one cell per pair, each cell
// a paired Exact Match difference marked as a win, a loss or a tie.
//
// The test in each cell is McNemar's on the questions the two systems
// disagree on, the same test cmd/compare runs, but the multiple-comparison
// correction is different and stricter: Holm across EVERY cell of the table
// at once, because the claim the table supports is "wins W, loses L, ties T
// over P pairs", and that is one family of tests. A dozen ties that would be
// wins under a per-pair correction are the price of being able to say so.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type record struct {
	Index    int     `json:"index"`
	Question string  `json:"question"`
	EM       float64 `json:"em"`
	LLMCalls int64   `json:"llm_calls"`
}

// cell is one (baseline, set) comparison.
type cell struct {
	baseline, set  string
	n              int
	meanA, meanB   float64
	diff, lo, hi   float64
	p, pAdjusted   float64
	callsA, callsB float64
	verdict        string // win | loss | tie
	missing        bool
}

// Display names. A baseline that is not listed is shown under its slug.
var baselineNames = map[string]string{
	"closedbook": "ClosedBook (no retrieval)",
	"bm25":       "BM25",
	"naive":      "NaiveRAG, 5 passages",
	"naive10":    "NaiveRAG, 10 passages",
	"naive12":    "NaiveRAG, 12 passages",
	"hybrid":     "Hybrid (dense + BM25)",
	"hyde":       "HyDE",
	"rerank":     "Rerank (cross-encoder)",
	"neighbour":  "Neighbour expansion",
	"crag":       "CRAG (offline), 5 passages",
	"crag10":     "CRAG (offline), 10 passages",
	"ircot":      "IRCoT",
	"adaptive":   "Adaptive-RAG",
	"cascade":    "Cascade",
	"cascade-lp": "Cascade + confidence",
}

var setNames = map[string]string{
	"nq":       "NQ",
	"triviaqa": "TriviaQA",
	"hotpotqa": "HotpotQA",
	"2wiki":    "2Wiki",
	"musique":  "MuSiQue",
}

func main() {
	var (
		runsDir   = flag.String("runs", "runs", "directory holding the -dump files, named <architecture>-<set>.jsonl")
		proposed  = flag.String("proposed", "cascade", "architecture slug of the proposed system")
		baselines = flag.String("baselines", "closedbook,bm25,naive,naive10,naive12,hybrid,hyde,rerank,neighbour,crag,crag10,ircot,adaptive", "comma-separated baseline slugs, in table order")
		sets      = flag.String("sets", "nq,triviaqa,hotpotqa,2wiki,musique", "comma-separated question-set slugs, in column order")
		alpha     = flag.Float64("alpha", 0.05, "significance level after the Holm adjustment")
		resamples = flag.Int("bootstrap", 2000, "bootstrap resamples for each cell's confidence interval")
		seed      = flag.Int64("seed", 42, "bootstrap seed")
		texOut    = flag.String("tex-out", "", "path of a .tex file to write the table to - optional")
		name      = flag.String("name", "Result", "prefix of the generated .tex commands")
	)
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	var cells []*cell
	for _, b := range strings.Split(*baselines, ",") {
		for _, s := range strings.Split(*sets, ",") {
			c := &cell{baseline: strings.TrimSpace(b), set: strings.TrimSpace(s)}
			cells = append(cells, c)
			a, okA := load(filepath.Join(*runsDir, c.baseline+"-"+c.set+".jsonl"))
			p, okB := load(filepath.Join(*runsDir, *proposed+"-"+c.set+".jsonl"))
			if !okA || !okB {
				c.missing = true
				continue
			}
			compare(c, a, p, *resamples, rng)
		}
	}
	holm(cells)
	wins, losses, ties := 0, 0, 0
	for _, c := range cells {
		if c.missing {
			continue
		}
		switch {
		case c.pAdjusted >= *alpha:
			c.verdict = "tie"
			ties++
		case c.diff > 0:
			c.verdict = "win"
			wins++
		default:
			c.verdict = "loss"
			losses++
		}
	}

	setList := strings.Split(*sets, ",")
	fmt.Printf("%-28s", "baseline (calls/q)")
	for _, s := range setList {
		fmt.Printf("%22s", setNames[s])
	}
	fmt.Println()
	for _, b := range strings.Split(*baselines, ",") {
		row := rowCells(cells, b)
		fmt.Printf("%-28s", fmt.Sprintf("%s (%s)", baselineNames[b], callsLabel(row, true)))
		for _, c := range row {
			if c.missing {
				fmt.Printf("%22s", "-")
				continue
			}
			fmt.Printf("%22s", fmt.Sprintf("%-4s %+.4f p=%s", c.verdict, c.diff, formatP(c.pAdjusted)))
		}
		fmt.Println()
	}
	fmt.Printf("%-28s", fmt.Sprintf("%s calls/q", baselineNames[*proposed]))
	for _, s := range setList {
		fmt.Printf("%22s", proposedCalls(cells, s))
	}
	fmt.Printf("\n\n%d pairs: %d wins, %d losses, %d ties (Holm over all %d tests, alpha %.2f)\n", wins+losses+ties, wins, losses, ties, wins+losses+ties, *alpha)

	if *texOut != "" {
		if err := writeTex(*texOut, *name, *proposed, strings.Split(*baselines, ","), setList, cells, wins, losses, ties); err != nil {
			log.Fatalf("writing tex: %v", err)
		}
		log.Printf("written to %s", *texOut)
	}
}

func rowCells(cells []*cell, baseline string) []*cell {
	var out []*cell
	for _, c := range cells {
		if c.baseline == baseline {
			out = append(out, c)
		}
	}
	return out
}

// callsLabel summarises a baseline's generation calls per question over the
// sets it ran on, as one number when they agree and a range otherwise.
func callsLabel(row []*cell, baseline bool) string {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, c := range row {
		if c.missing {
			continue
		}
		v := c.callsB
		if baseline {
			v = c.callsA
		}
		lo, hi = min(lo, v), max(hi, v)
	}
	if math.IsInf(lo, 1) {
		return "-"
	}
	if hi-lo < 0.05 {
		return fmt.Sprintf("%.1f", (lo+hi)/2)
	}
	return fmt.Sprintf("%.1f-%.1f", lo, hi)
}

func proposedCalls(cells []*cell, set string) string {
	for _, c := range cells {
		if c.set == set && !c.missing {
			return fmt.Sprintf("%.2f", c.callsB)
		}
	}
	return "-"
}

func compare(c *cell, a, b map[key]record, resamples int, rng *rand.Rand) {
	var va, vb, diffs []float64
	var callsA, callsB float64
	for k, ra := range a {
		rb, ok := b[k]
		if !ok {
			continue
		}
		va = append(va, ra.EM)
		vb = append(vb, rb.EM)
		diffs = append(diffs, rb.EM-ra.EM)
		callsA += float64(ra.LLMCalls)
		callsB += float64(rb.LLMCalls)
	}
	c.n = len(va)
	if c.n == 0 {
		c.missing = true
		return
	}
	n := float64(c.n)
	c.meanA, c.meanB = mean(va), mean(vb)
	c.diff = c.meanB - c.meanA
	c.callsA, c.callsB = callsA/n, callsB/n
	c.lo, c.hi = bootstrapCI(diffs, resamples, rng)
	c.p = mcnemar(va, vb)
}

// holm applies the Holm-Bonferroni step-down correction over every cell.
func holm(cells []*cell) {
	var tested []*cell
	for _, c := range cells {
		if !c.missing {
			tested = append(tested, c)
		}
	}
	slices.SortStableFunc(tested, func(x, y *cell) int {
		switch {
		case x.p < y.p:
			return -1
		case x.p > y.p:
			return 1
		}
		return 0
	})
	m := len(tested)
	running := 0.0
	for i, c := range tested {
		running = max(running, float64(m-i)*c.p)
		c.pAdjusted = min(running, 1)
	}
}

func writeTex(path, name, proposed string, baselines, sets []string, cells []*cell, wins, losses, ties int) error {
	id := texSafeID(name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/result - do not edit by hand\n")
	fmt.Fprintf(&b, "%% A cell is the paired Exact Match difference (%s minus baseline). Bold: %s is significantly better;\n", baselineNames[proposed], baselineNames[proposed])
	fmt.Fprintf(&b, "%% underlined: significantly worse; plain: no significant difference. McNemar, Holm over every cell.\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sPairs}{%d}\n", id, wins+losses+ties)
	fmt.Fprintf(&b, "\\newcommand{\\%sWins}{%d}\n", id, wins)
	fmt.Fprintf(&b, "\\newcommand{\\%sLosses}{%d}\n", id, losses)
	fmt.Fprintf(&b, "\\newcommand{\\%sTies}{%d}\n", id, ties)
	fmt.Fprintf(&b, "\\newcommand{\\%sTable}{%%\n", id)
	fmt.Fprintf(&b, "\\begin{tabular}{l%s}\n\\toprule\n", strings.Repeat("r", len(sets)))
	fmt.Fprintf(&b, "Baseline (calls/q)")
	for _, s := range sets {
		fmt.Fprintf(&b, " & %s", setNames[s])
	}
	fmt.Fprintf(&b, " \\\\\n\\midrule\n")
	for _, bl := range baselines {
		row := rowCells(cells, bl)
		fmt.Fprintf(&b, "%s (%s)", baselineNames[bl], callsLabel(row, true))
		for _, c := range row {
			switch {
			case c.missing:
				fmt.Fprintf(&b, " & --")
			case c.verdict == "win":
				fmt.Fprintf(&b, " & \\textbf{%+.3f}", c.diff)
			case c.verdict == "loss":
				fmt.Fprintf(&b, " & \\underline{%+.3f}", c.diff)
			default:
				fmt.Fprintf(&b, " & %+.3f", c.diff)
			}
		}
		fmt.Fprintf(&b, " \\\\\n")
	}
	fmt.Fprintf(&b, "\\midrule\n%s calls/q", baselineNames[proposed])
	for _, s := range sets {
		fmt.Fprintf(&b, " & %s", proposedCalls(cells, s))
	}
	fmt.Fprintf(&b, " \\\\\n\\bottomrule\n\\end{tabular}}\n")

	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
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

func mean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func bootstrapCI(values []float64, resamples int, rng *rand.Rand) (lo, hi float64) {
	n := len(values)
	if n == 0 || resamples <= 0 {
		return 0, 0
	}
	means := make([]float64, resamples)
	for b := range resamples {
		s := 0.0
		for range n {
			s += values[rng.Intn(n)]
		}
		means[b] = s / float64(n)
	}
	slices.Sort(means)
	return means[resamples/40], means[resamples-resamples/40-1]
}

// mcnemar is the same test cmd/compare runs: normal approximation with
// continuity correction over the discordant pairs.
func mcnemar(a, b []float64) float64 {
	var onlyA, onlyB int
	for i := range a {
		switch {
		case a[i] >= 0.5 && b[i] < 0.5:
			onlyA++
		case a[i] < 0.5 && b[i] >= 0.5:
			onlyB++
		}
	}
	n := onlyA + onlyB
	if n == 0 {
		return 1
	}
	corrected := max(math.Abs(float64(onlyA-onlyB))-1, 0)
	chi := corrected * corrected / float64(n)
	return 2 * (1 - 0.5*math.Erfc(-math.Sqrt(chi)/math.Sqrt2))
}

func formatP(p float64) string {
	if p < 1e-4 {
		return "<0.0001"
	}
	return fmt.Sprintf("%.4f", p)
}

type key struct {
	index    int
	question string
}

// load reads a dump keyed by (index, question); ok is false when the file
// does not exist, which is how a baseline that was not run on a set shows up
// as "--" rather than as an error.
func load(path string) (map[key]record, bool) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false
		}
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
		out[key{r.Index, r.Question}] = r
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("reading %s: %v", path, err)
	}
	return out, true
}
