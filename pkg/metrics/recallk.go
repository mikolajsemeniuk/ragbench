package metrics

// RecallAtK returns the fraction of distinct relevant items that appear within
// the first k positions of the retrieved list (ordered by decreasing
// relevance). A retrieved list longer than k is trimmed to k.
//
// It is generic over the item type because what counts as "the relevant
// document" depends on the dataset: passage IDs for a corpus with per-passage
// annotations, article titles for the multi-hop sets whose gold annotation is
// a list of supporting article titles.
//
// Distinct matters. A Wikipedia article is split into many passages that share
// a title, so counting every retrieved hit separately would let one article
// satisfy the recall of several, and would let a single question score above
// its own ceiling.
func RecallAtK[T comparable](retrieved []T, relevant []T, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	if k < len(retrieved) {
		retrieved = retrieved[:k]
	}

	relevantSet := make(map[T]struct{}, len(relevant))
	for _, item := range relevant {
		relevantSet[item] = struct{}{}
	}

	found := make(map[T]struct{}, len(relevantSet))
	for _, item := range retrieved {
		if _, ok := relevantSet[item]; ok {
			found[item] = struct{}{}
		}
	}
	return float64(len(found)) / float64(len(relevantSet))
}
