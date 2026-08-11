// Package metrics dostarcza metryki oceny jakości odpowiedzi QA (Exact
// Match, F1) używane przez cmd/bench do ewaluacji baseline'u i przyszłych
// architektur RAG.
package metrics

import "strings"

// Normalize ujednolica tekst przed porównaniem: lowercase, bez interpunkcji,
// bez wielokrotnych spacji (standardowa normalizacja EM/F1 dla QA).
func Normalize(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if strings.ContainsRune(".,!?;:'\"()[]{}", r) {
			continue
		}

		b.WriteRune(r)
	}

	return strings.Join(strings.Fields(b.String()), " ")
}
