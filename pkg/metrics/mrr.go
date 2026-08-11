package metrics

// ReciprocalRank zwraca 1/rank pierwszego trafnego (relevant) dokumentu na
// liście retrieved (indeksowanej od 1), albo 0, jeśli żaden trafny dokument
// nie został znaleziony. Uśrednienie ReciprocalRank po zbiorze zapytań daje
// MRR (Mean Reciprocal Rank).
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
