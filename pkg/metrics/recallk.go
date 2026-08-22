package metrics

// RecallAtK returns the fraction of relevant documents that appear within
// the first k positions of the retrieved list (ordered by decreasing
// relevance). A retrieved list longer than k is trimmed to k.
func RecallAtK(retrieved []uint64, relevant []uint64, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	if k < len(retrieved) {
		retrieved = retrieved[:k]
	}

	relevantSet := make(map[uint64]struct{}, len(relevant))
	for _, id := range relevant {
		relevantSet[id] = struct{}{}
	}

	hits := 0
	for _, id := range retrieved {
		if _, ok := relevantSet[id]; ok {
			hits++
		}
	}

	return float64(hits) / float64(len(relevant))
}
