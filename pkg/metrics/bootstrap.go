package metrics

import (
	"math/rand"
	"slices"
)

// BootstrapCI returns the 95% percentile bootstrap confidence interval for the
// mean of values, using the given number of resamples.
//
// Why bootstrap rather than a closed-form interval: Exact Match and
// answer-in-context are proportions, so a binomial interval would do, but F1
// and Recall@K are per-question fractions with no analytic form. Resampling
// handles every metric with one method, which keeps the intervals in a results
// table mutually comparable.
//
// rng is passed in so a run with a fixed -seed reproduces its intervals
// exactly.
func BootstrapCI(values []float64, resamples int, rng *rand.Rand) (lo, hi float64) {
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

	return means[percentileIndex(resamples, 0.025)], means[percentileIndex(resamples, 0.975)]
}

func percentileIndex(n int, p float64) int {
	i := int(p * float64(n))
	return min(max(i, 0), n-1)
}
