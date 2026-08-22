package metrics

// ReciprocalRank returns 1/rank of the first relevant document in the
// retrieved list (1-indexed), or 0 if no relevant document was retrieved.
// Averaging ReciprocalRank over a set of queries yields MRR (Mean Reciprocal
// Rank).
func ReciprocalRank(retrieved []uint64, relevant []uint64) float64 {
	if len(relevant) == 0 {
		return 0
	}

	relevantSet := make(map[uint64]struct{}, len(relevant))
	for _, id := range relevant {
		relevantSet[id] = struct{}{}
	}

	for i, id := range retrieved {
		if _, ok := relevantSet[id]; ok {
			return 1 / float64(i+1)
		}
	}

	return 0
}
