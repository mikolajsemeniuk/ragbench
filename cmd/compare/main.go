// Command compare takes the per-question dumps of two cmd/bench runs and
// reports whether the difference between them is real.
//
// Comparing two means tells you which number is bigger, not whether the gap
// would survive a different sample of questions. Because both systems answered
// the *same* questions, the comparison can be paired, which removes question
// difficulty from the variance and is far more sensitive than comparing two
// independent intervals: two overlapping confidence intervals can still hide a
// consistent per-question win.
//
//   - Exact Match is binary, so the paired test is McNemar's, computed on the
//     questions where exactly one of the two systems is correct. Those
//     discordant pairs carry all the information; questions both got right or
//     both got wrong say nothing about which is better.
//   - F1 and the continuous metrics use the Wilcoxon signed-rank test over the
//     per-question differences.
//   - The size of the difference is reported as a paired bootstrap confidence
//     interval, because a p-value says whether an effect exists, not whether
//     it is large enough to matter.
//   - Every p-value is Holm-adjusted across the metrics reported here. A dozen
//     tests on one pair of runs will produce a p below 0.05 by chance alone,
//     and the effects being measured are around one percentage point. Exact
//     Match is the primary endpoint; the rest are descriptive even after
//     adjustment.
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
	F1       float64 `json:"f1"`

	Abstained bool `json:"abstained"`

	InCtx           float64 `json:"answer_in_context"`
	InCtxApplicable bool    `json:"answer_in_context_applicable"`
	HasGold         bool    `json:"has_gold"`
	Recall          float64 `json:"recall_in_context"`
	RR              float64 `json:"reciprocal_rank"`

	Passages     int   `json:"passages_in_context"`
	LLMCalls     int64 `json:"llm_calls"`
	PromptTokens int64 `json:"prompt_tokens"`

	LatencyS float64 `json:"latency_seconds"`
}

// row is one metric of the comparison. Everything is computed before anything
// is printed, because the Holm adjustment needs the whole family of p-values.
type row struct {
	name  string
	get   func(record) float64
	keep  func(a, b record) bool
	test  string // mcnemar | wilcoxon
	notes string

	n            int
	meanA, meanB float64
	lo, hi       float64
	p            float64
	pAdjusted    float64
	detail       string
	empty        bool
}

func main() {
	var (
		pathA     = flag.String("a", "", "path to the baseline run's -dump file (required)")
		pathB     = flag.String("b", "", "path to the compared run's -dump file (required)")
		nameA     = flag.String("name-a", "A", "label for the baseline run")
		nameB     = flag.String("name-b", "B", "label for the compared run")
		resamples = flag.Int("bootstrap", 10000, "bootstrap resamples for the confidence interval of the difference")
		seed      = flag.Int64("seed", 42, "bootstrap seed, so the reported interval is reproducible")
		texOut    = flag.String("tex-out", "", "path of a .tex file to write the differences and their intervals to - optional")
		name      = flag.String("name", "", "name used in the generated .tex commands (defaults to name-a vs name-b)")
	)
	flag.Parse()

	if *pathA == "" || *pathB == "" {
		log.Fatal("both -a and -b are required")
	}

	a, err := load(*pathA)
	if err != nil {
		log.Fatalf("reading %s: %v", *pathA, err)
	}
	b, err := load(*pathB)
	if err != nil {
		log.Fatalf("reading %s: %v", *pathB, err)
	}

	pa, pb, byIndex := pair(a, b)
	if len(pa) == 0 {
		log.Fatal("the two dumps share no questions - are they from the same dataset?")
	}

	fmt.Printf("--- %s vs %s ---\n", *nameA, *nameB)
	fmt.Printf("questions in %s: %d, in %s: %d, paired: %d (%d matched on index, the rest on question text)\n\n", *nameA, len(a), *nameB, len(b), len(pa), byIndex)
	if len(pa) < len(a) || len(pa) < len(b) {
		fmt.Printf("NOTE: %d/%d and %d/%d questions matched; unmatched ones are excluded from every figure below.\n\n", len(pa), len(a), len(pa), len(b))
	}

	both := func(a, b record) bool { return true }
	gold := func(a, b record) bool { return a.HasGold && b.HasGold }
	rows := []*row{
		{name: "Exact Match", get: func(r record) float64 { return r.EM }, keep: both, test: "mcnemar", notes: "primary endpoint"},
		{name: "F1", get: func(r record) float64 { return r.F1 }, keep: both, test: "wilcoxon"},
		{name: "Abstention", get: func(r record) float64 { return boolean(r.Abstained) }, keep: both, test: "mcnemar"},
		// Exact Match restricted to the questions neither system declined.
		// Without it a system that refuses more often looks worse on Exact
		// Match than one that guesses, which is a difference in behaviour
		// rather than in knowledge - the effect that reverses the
		// ClosedBook/NaiveRAG ordering on 2WikiMultihopQA.
		{name: "EM (both answered)", get: func(r record) float64 { return r.EM }, keep: func(x, y record) bool { return !x.Abstained && !y.Abstained }, test: "mcnemar"},
		{name: "Answer in context", get: func(r record) float64 { return r.InCtx }, keep: func(x, y record) bool { return x.InCtxApplicable && y.InCtxApplicable }, test: "mcnemar"},
		{name: "Recall in context", get: func(r record) float64 { return r.Recall }, keep: gold, test: "wilcoxon"},
		{name: "MRR", get: func(r record) float64 { return r.RR }, keep: gold, test: "wilcoxon"},
		{name: "Passages in ctx", get: func(r record) float64 { return float64(r.Passages) }, keep: both, test: "wilcoxon"},
		{name: "LLM calls", get: func(r record) float64 { return float64(r.LLMCalls) }, keep: both, test: "wilcoxon"},
		{name: "Prompt tokens", get: func(r record) float64 { return float64(r.PromptTokens) }, keep: both, test: "wilcoxon"},
		{name: "Latency (s)", get: func(r record) float64 { return r.LatencyS }, keep: both, test: "wilcoxon"},
	}

	rng := rand.New(rand.NewSource(*seed))
	for _, r := range rows {
		compute(r, pa, pb, *resamples, rng)
	}
	holm(rows)

	fmt.Printf("%-20s %10s %10s %10s   %-24s %-10s %-10s %s\n", "metric", *nameA, *nameB, "diff", "95% CI of diff", "p", "p (Holm)", "test")
	for _, r := range rows {
		if r.empty {
			fmt.Printf("%-20s %10s %10s %10s   %-24s %-10s %-10s %s\n", r.name, "-", "-", "-", "not applicable", "-", "-", "-")
			continue
		}
		note := r.detail
		if r.notes != "" {
			note += "  [" + r.notes + "]"
		}
		fmt.Printf("%-20s %10.4f %10.4f %+10.4f   [%+.4f, %+.4f]   %-10s %-10s %s\n",
			r.name, r.meanA, r.meanB, r.meanB-r.meanA, r.lo, r.hi, formatP(r.p), formatP(r.pAdjusted), note)
	}

	fmt.Printf("\nA 95%% CI of the difference that excludes 0 means the gap survives resampling.\n")
	fmt.Printf("Paired tests use only the questions where the two systems disagree.\n")
	fmt.Printf("p (Holm) is adjusted across the %d metrics above; read Exact Match as the primary endpoint and the rest as descriptive.\n", countTested(rows))

	if *texOut != "" {
		label := *name
		if label == "" {
			label = *nameA + "Vs" + *nameB
		}
		if err := writeTex(*texOut, label, len(pa), rows); err != nil {
			log.Fatalf("writing tex: %v", err)
		}
		log.Printf("written to %s", *texOut)
	}
}

func boolean(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

// pair matches the two runs question by question.
//
// The key is (index, question), falling back to the question text alone. Index
// alone breaks when one run skipped a failed question; question text alone
// breaks on a dataset with duplicate questions - musique_dev has five, and a
// map keyed on text silently keeps only the last of each. Trying the exact
// pair first and the text second is correct for both.
func pair(a, b []record) (pa, pb []record, matchedOnIndex int) {
	type key struct {
		index    int
		question string
	}
	byPair := make(map[key]int, len(b))
	byQuestion := make(map[string]int, len(b))
	for i, r := range b {
		byPair[key{r.Index, r.Question}] = i
		byQuestion[r.Question] = i
	}

	used := make(map[int]struct{}, len(b))
	for _, ra := range a {
		j, ok := byPair[key{ra.Index, ra.Question}]
		if ok {
			matchedOnIndex++
		} else if j, ok = byQuestion[ra.Question]; !ok {
			continue
		}
		if _, taken := used[j]; taken {
			continue
		}
		used[j] = struct{}{}
		pa = append(pa, ra)
		pb = append(pb, b[j])
	}
	return pa, pb, matchedOnIndex
}

func compute(r *row, pa, pb []record, resamples int, rng *rand.Rand) {
	var va, vb []float64
	for i := range pa {
		if !r.keep(pa[i], pb[i]) {
			continue
		}
		va = append(va, r.get(pa[i]))
		vb = append(vb, r.get(pb[i]))
	}
	if len(va) == 0 {
		r.empty = true
		return
	}

	r.n = len(va)
	r.meanA, r.meanB = mean(va), mean(vb)
	diffs := make([]float64, len(va))
	for i := range va {
		diffs[i] = vb[i] - va[i]
	}
	r.lo, r.hi = bootstrapCI(diffs, resamples, rng)

	switch r.test {
	case "mcnemar":
		r.p, r.detail = mcnemar(va, vb)
	case "wilcoxon":
		r.p, r.detail = wilcoxon(diffs)
	default:
		r.p = math.NaN()
	}
	if r.n != len(pa) {
		r.detail = fmt.Sprintf("n=%d  %s", r.n, r.detail)
	}
}

// holm applies the Holm-Bonferroni step-down correction over the metrics that
// produced a p-value. It controls the family-wise error rate without assuming
// the tests are independent, which they are not - Recall, MRR and
// answer-in-context all move together.
func holm(rows []*row) {
	var tested []*row
	for _, r := range rows {
		if !r.empty && !math.IsNaN(r.p) {
			tested = append(tested, r)
		}
	}
	slices.SortStableFunc(tested, func(x, y *row) int {
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
	for i, r := range tested {
		adjusted := float64(m-i) * r.p
		running = max(running, adjusted)
		r.pAdjusted = min(running, 1)
	}
}

func countTested(rows []*row) int {
	n := 0
	for _, r := range rows {
		if !r.empty && !math.IsNaN(r.p) {
			n++
		}
	}
	return n
}

func writeTex(path, label string, paired int, rows []*row) error {
	id := texSafeID(label)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/compare - do not edit by hand\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sPaired}{%d}\n", id, paired)
	for _, r := range rows {
		if r.empty {
			continue
		}
		key := texSafeID(r.name)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sA}{%.4f}\n", id, key, r.meanA)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sB}{%.4f}\n", id, key, r.meanB)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sDiff}{%+.4f}\n", id, key, r.meanB-r.meanA)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sCILow}{%+.4f}\n", id, key, r.lo)
		fmt.Fprintf(&b, "\\newcommand{\\%s%sCIHigh}{%+.4f}\n", id, key, r.hi)
		if !math.IsNaN(r.p) {
			fmt.Fprintf(&b, "\\newcommand{\\%s%sP}{%s}\n", id, key, formatP(r.p))
			fmt.Fprintf(&b, "\\newcommand{\\%s%sPHolm}{%s}\n", id, key, formatP(r.pAdjusted))
		}
	}

	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create dir: %w", err)
		}
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

func load(path string) ([]record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []record
	checked := false
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		if !checked {
			if err := checkSchema(line); err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			checked = true
		}
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, fmt.Errorf("parse line: %w", err)
		}
		out = append(out, r)
	}
	return out, scanner.Err()
}

// checkSchema refuses a dump written before the metrics were fixed.
//
// The failure it prevents is silent rather than loud. An old dump has no
// "abstained" and no "recall_in_context" - the retrieval field used to be
// called "recall_at_k" - so every one of those metrics would decode as zero
// for both runs and be reported as a real, perfectly tied comparison. Refusing
// to read the file is the only honest response; the run has to be redone.
func checkSchema(line []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(line, &fields); err != nil {
		return fmt.Errorf("parse first record: %w", err)
	}
	for _, required := range []string{"abstained", "answer_in_context_applicable", "llm_calls"} {
		if _, ok := fields[required]; !ok {
			return fmt.Errorf("this dump predates the current metrics (no %q field). Abstention, answer-in-context applicability and the retrieval metrics would all read as zero. Re-run cmd/bench to regenerate it", required)
		}
	}
	return nil
}

func mean(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x
	}
	return sum / float64(len(v))
}

func bootstrapCI(values []float64, resamples int, rng *rand.Rand) (lo, hi float64) {
	n := len(values)
	if n == 0 || resamples <= 0 {
		return 0, 0
	}
	means := make([]float64, resamples)
	for b := range resamples {
		sum := 0.0
		for range n {
			sum += values[rng.Intn(n)]
		}
		means[b] = sum / float64(n)
	}
	slices.Sort(means)
	return means[resamples/40], means[resamples-resamples/40-1]
}

// mcnemar runs McNemar's test on paired binary outcomes, using the normal
// approximation with continuity correction. onlyA and onlyB are the discordant
// counts: questions only A got right, and questions only B got right.
func mcnemar(a, b []float64) (float64, string) {
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
		return 1, "identical on every question"
	}

	// The continuity correction cannot make the statistic negative: with equal
	// discordant counts the evidence for a difference is nil, not slightly
	// against it.
	corrected := max(math.Abs(float64(onlyA-onlyB))-1, 0)
	chi := corrected * corrected / float64(n)
	return chiSquareP(chi), fmt.Sprintf("McNemar (only A: %d, only B: %d)", onlyA, onlyB)
}

// wilcoxon runs the Wilcoxon signed-rank test on paired differences, using the
// normal approximation with tie-corrected ranks. Suitable here because the
// sample sizes are in the thousands.
func wilcoxon(diffs []float64) (float64, string) {
	type entry struct {
		abs  float64
		sign float64
	}
	var nonZero []entry
	for _, d := range diffs {
		if d != 0 {
			sign := 1.0
			if d < 0 {
				sign = -1
			}
			nonZero = append(nonZero, entry{abs: math.Abs(d), sign: sign})
		}
	}
	n := len(nonZero)
	if n == 0 {
		return 1, "identical on every question"
	}
	slices.SortFunc(nonZero, func(x, y entry) int {
		switch {
		case x.abs < y.abs:
			return -1
		case x.abs > y.abs:
			return 1
		}
		return 0
	})

	// Average ranks within ties, otherwise ties inflate the statistic.
	ranks := make([]float64, n)
	for i := 0; i < n; {
		j := i
		for j+1 < n && nonZero[j+1].abs == nonZero[i].abs {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			ranks[k] = avg
		}
		i = j + 1
	}

	var wPlus float64
	for i, e := range nonZero {
		if e.sign > 0 {
			wPlus += ranks[i]
		}
	}
	nf := float64(n)
	meanW := nf * (nf + 1) / 4
	sdW := math.Sqrt(nf * (nf + 1) * (2*nf + 1) / 24)
	if sdW == 0 {
		return 1, "degenerate"
	}
	z := (wPlus - meanW) / sdW
	return 2 * (1 - normalCDF(math.Abs(z))), fmt.Sprintf("Wilcoxon (n=%d non-tied)", n)
}

func chiSquareP(chi float64) float64 {
	// One degree of freedom: P(X > chi) = 2 * (1 - Phi(sqrt(chi))).
	return 2 * (1 - normalCDF(math.Sqrt(chi)))
}

func normalCDF(z float64) float64 {
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}

func formatP(p float64) string {
	if math.IsNaN(p) {
		return "n/a"
	}
	if p < 1e-4 {
		return "<0.0001"
	}
	return fmt.Sprintf("%.4f", p)
}
