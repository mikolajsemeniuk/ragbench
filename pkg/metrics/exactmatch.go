package metrics

// ExactMatch zwraca 1, jeśli znormalizowana odpowiedź jest identyczna z jedną
// ze złotych odpowiedzi, w przeciwnym razie 0.
func ExactMatch(answer string, golden []string) float64 {
	na := Normalize(answer)
	for _, g := range golden {
		if na == Normalize(g) {
			return 1
		}
	}

	return 0
}
