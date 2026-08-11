package metrics

// TokenF1 liczy F1 na poziomie tokenów między przewidywaną a złotą sekwencją
// tokenów (standardowa metryka dla SQuAD/NQ/HotpotQA).
func TokenF1(pred, gold []string) float64 {
	if len(pred) == 0 || len(gold) == 0 {
		if len(pred) == len(gold) {
			return 1
		}
		return 0
	}

	counts := make(map[string]int, len(gold))
	for _, t := range gold {
		counts[t]++
	}

	common := 0
	for _, t := range pred {
		if counts[t] > 0 {
			common++
			counts[t]--
		}
	}
	if common == 0 {
		return 0
	}

	precision := float64(common) / float64(len(pred))
	recall := float64(common) / float64(len(gold))
	return 2 * precision * recall / (precision + recall)
}
