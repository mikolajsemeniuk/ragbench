package metrics

// ExactMatch returns 1 if the normalised answer is identical to one of the
// gold answers, and 0 otherwise.
func ExactMatch(answer string, golden []string) float64 {
	na := Normalize(answer)
	for _, g := range golden {
		if na == Normalize(g) {
			return 1
		}
	}

	return 0
}
