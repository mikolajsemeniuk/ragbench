// Command compare takes the per-question dumps of two cmd/bench runs and
// reports whether the difference between them is real.
//
// Comparing two means tells you which number is bigger, not whether the gap
// would survive a different sample of questions. Because both systems answered
// the *same* questions, the comparison can be paired, which removes
// question difficulty from the variance and is far more sensitive than
// comparing two independent intervals: two overlapping confidence intervals
// can still hide a consistent per-question win.
//
//   - Exact Match is binary, so the paired test is McNemar's, computed on the
//     questions where exactly one of the two systems is correct. Those
//     discordant pairs carry all the information; questions both got right or
//     both got wrong say nothing about which is better.
//   - F1 is continuous, so the paired test is Wilcoxon signed-rank over the
//     per-question differences.
//   - The size of the difference is reported as a paired bootstrap confidence
//     interval, because a p-value says whether an effect exists, not whether
//     it is large enough to matter.
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
	"slices"
)

type record struct {
	Index    int     `json:"index"`
	Question string  `json:"question"`
	EM       float64 `json:"em"`
	F1       float64 `json:"f1"`
	InCtx    float64 `json:"answer_in_context"`
	HasGold  bool    `json:"has_gold"`
	Recall   float64 `json:"recall_at_k"`
	RR       float64 `json:"reciprocal_rank"`
	Passages int     `json:"passages_in_context"`
	LatencyS float64 `json:"latency_seconds"`
}

func main() {
	var (
		pathA     = flag.String("a", "", "path to the baseline run's -dump file (required)")
		pathB     = flag.String("b", "", "path to the compared run's -dump file (required)")
		nameA     = flag.String("name-a", "A", "label for the baseline run")
		nameB     = flag.String("name-b", "B", "label for the compared run")
		resamples = flag.Int("bootstrap", 10000, "bootstrap resamples for the confidence interval of the difference")
		seed      = flag.Int64("seed", 42, "bootstrap seed, so the reported interval is reproducible")
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

	// Pair by question text rather than by line number: a run that skipped a
	// failed question, or used a different -limit sample, would otherwise be
	// silently misaligned and every reported difference would be noise.
	byQuestion := make(map[string]int, len(b))
	for i, r := range b {
		byQuestion[r.Question] = i
	}
	var pa, pb []record
	for _, ra := range a {
		if j, ok := byQuestion[ra.Question]; ok {
			pa = append(pa, ra)
			pb = append(pb, b[j])
		}
	}
	if len(pa) == 0 {
		log.Fatal("the two dumps share no questions - are they from the same dataset?")
	}

	fmt.Printf("--- %s vs %s ---\n", *nameA, *nameB)
	fmt.Printf("questions in %s: %d, in %s: %d, paired: %d\n\n", *nameA, len(a), *nameB, len(b), len(pa))
	if len(pa) < len(a) || len(pa) < len(b) {
		fmt.Printf("NOTE: %d/%d and %d/%d questions matched; unmatched ones are excluded from every figure below.\n\n", len(pa), len(a), len(pa), len(b))
	}

	rng := rand.New(rand.NewSource(*seed))
	fmt.Printf("%-20s %10s %10s %10s   %-24s %s\n", "metric", *nameA, *nameB, "diff", "95% CI of diff", "paired test")

	report := func(name string, get func(record) float64, gated bool, test string) {
		var va, vb []float64
		for i := range pa {
			if gated && !(pa[i].HasGold && pb[i].HasGold) {
				continue
			}
			va = append(va, get(pa[i]))
			vb = append(vb, get(pb[i]))
		}
		if len(va) == 0 {
			fmt.Printf("%-20s %10s %10s %10s   %-24s %s\n", name, "-", "-", "-", "not annotated", "-")
			return
		}

		ma, mb := mean(va), mean(vb)
		diffs := make([]float64, len(va))
		for i := range va {
			diffs[i] = vb[i] - va[i]
		}
		lo, hi := bootstrapCI(diffs, *resamples, rng)

		var verdict string
		switch test {
		case "mcnemar":
			verdict = mcnemar(va, vb)
		case "wilcoxon":
			verdict = wilcoxon(diffs)
		}
		fmt.Printf("%-20s %10.4f %10.4f %+10.4f   [%+.4f, %+.4f]   %s\n", name, ma, mb, mb-ma, lo, hi, verdict)
	}

	report("Exact Match", func(r record) float64 { return r.EM }, false, "mcnemar")
	report("F1", func(r record) float64 { return r.F1 }, false, "wilcoxon")
	report("Answer in context", func(r record) float64 { return r.InCtx }, false, "mcnemar")
	report("Recall@K", func(r record) float64 { return r.Recall }, true, "wilcoxon")
	report("MRR", func(r record) float64 { return r.RR }, true, "wilcoxon")
	report("Passages in ctx", func(r record) float64 { return float64(r.Passages) }, false, "wilcoxon")
	report("Latency (s)", func(r record) float64 { return r.LatencyS }, false, "wilcoxon")

	fmt.Printf("\nA 95%% CI of the difference that excludes 0 means the gap survives resampling.\n")
	fmt.Printf("Paired tests use only the questions where the two systems disagree.\n")
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
// approximation with continuity correction. b and c are the discordant counts:
// questions only A got right, and questions only B got right.
func mcnemar(a, b []float64) string {
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
		return "identical on every question"
	}
	chi := math.Pow(math.Abs(float64(onlyA-onlyB))-1, 2) / float64(n)
	return fmt.Sprintf("McNemar p=%s (only A: %d, only B: %d)", formatP(chiSquareP(chi)), onlyA, onlyB)
}

// wilcoxon runs the Wilcoxon signed-rank test on paired differences, using the
// normal approximation with tie-corrected ranks. Suitable here because the
// sample sizes are in the thousands.
func wilcoxon(diffs []float64) string {
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
		return "identical on every question"
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
		return "degenerate"
	}
	z := (wPlus - meanW) / sdW
	return fmt.Sprintf("Wilcoxon p=%s (n=%d non-tied)", formatP(2*(1-normalCDF(math.Abs(z)))), n)
}

func chiSquareP(chi float64) float64 {
	// One degree of freedom: P(X > chi) = 2 * (1 - Phi(sqrt(chi))).
	return 2 * (1 - normalCDF(math.Sqrt(chi)))
}

func normalCDF(z float64) float64 {
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}

func formatP(p float64) string {
	if p < 1e-4 {
		return "<0.0001"
	}
	return fmt.Sprintf("%.4f", p)
}
