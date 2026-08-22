// Package metrics provides the QA answer-quality metrics (Exact Match, F1)
// used by cmd/bench to evaluate the baseline and further RAG architectures.
package metrics

import "strings"

// Normalize canonicalises text before comparison: lowercase, punctuation
// stripped, whitespace collapsed - the standard EM/F1 normalisation for QA.
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
