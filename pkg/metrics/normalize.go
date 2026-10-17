// Package metrics provides the QA answer-quality metrics (Exact Match, F1)
// used by cmd/bench to evaluate the baseline and further RAG architectures.
package metrics

import (
	"strings"
	"unicode"
)

// Normalize canonicalises text before comparison: lowercase, punctuation
// stripped, articles dropped, whitespace collapsed.
//
// Dropping "a", "an" and "the" is the part that is easy to miss and easy for a
// reviewer to spot. It is the standard SQuAD/NQ normalisation that every work
// this benchmark compares against applies, and without it "the White House"
// and "White House" count as different answers.
func Normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsPunct(r) {
			continue
		}
		b.WriteRune(r)
	}

	words := strings.Fields(b.String())
	kept := words[:0]
	for _, w := range words {
		if w == "a" || w == "an" || w == "the" {
			continue
		}
		kept = append(kept, w)
	}
	return strings.Join(kept, " ")
}
