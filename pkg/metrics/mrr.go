package metrics

// ReciprocalRank returns 1/rank of the first relevant item in the retrieved
// list (1-indexed), or 0 if no relevant item was retrieved. Averaging
// ReciprocalRank over a set of queries yields MRR (Mean Reciprocal Rank).
func ReciprocalRank[T comparable](retrieved []T, relevant []T) float64 {
	if len(relevant) == 0 {
		return 0
	}

	relevantSet := make(map[T]struct{}, len(relevant))
	for _, item := range relevant {
		relevantSet[item] = struct{}{}
	}

	for i, item := range retrieved {
		if _, ok := relevantSet[item]; ok {
			return 1 / float64(i+1)
		}
	}
	return 0
}
