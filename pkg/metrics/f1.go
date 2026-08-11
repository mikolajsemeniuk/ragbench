package metrics

import "strings"

// F1Score liczy token-level F1 między odpowiedzią a najlepiej pasującą złotą
// odpowiedzią.
func F1Score(answer string, golden []string) float64 {
	predTokens := strings.Fields(Normalize(answer))
	best := 0.0
	for _, g := range golden {
		goldTokens := strings.Fields(Normalize(g))
		f1 := TokenF1(predTokens, goldTokens)
		if f1 > best {
			best = f1
		}
	}
	return best
}
