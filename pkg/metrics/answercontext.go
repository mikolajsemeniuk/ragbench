package metrics

import "strings"

// AnswerInContext returns 1 if any retrieved passage states one of the gold
// answers, and 0 otherwise.
//
// This is the retrieval metric for datasets that annotate no gold documents at
// all - NaturalQuestions and TriviaQA are open-domain question sets with no
// supporting-passage labels, so title-based Recall@K cannot be computed for
// them. It answers the question that actually matters for a RAG pipeline: did
// retrieval put the information the generator needs in front of it?
//
// The match is on whole tokens, not on characters. A raw substring test looks
// harmless and is not: measured on the 2WikiMultihopQA baseline it scored
// 0.9740 for questions whose gold answer is "no", because "no" occurs inside
// "not", "now", "north" and "known", against 0.0369 for questions whose gold
// answer is "yes". Short answers in general were inflated the same way -
// 0.7759 for gold answers of at most three characters against 0.3490 for the
// rest.
//
// It remains a proxy, not ground truth: a passage can state the answer
// incidentally, without supporting the question. Reported alongside Recall@K
// rather than instead of it.
func AnswerInContext(passages []string, golden []string) float64 {
	var golds [][]string
	for _, g := range golden {
		if tokens := strings.Fields(Normalize(g)); len(tokens) > 0 {
			golds = append(golds, tokens)
		}
	}
	if len(golds) == 0 {
		return 0
	}

	for _, p := range passages {
		tokens := strings.Fields(Normalize(p))
		for _, g := range golds {
			if containsSequence(tokens, g) {
				return 1
			}
		}
	}
	return 0
}

// AnswerInContextApplicable reports whether answer-in-context means anything
// for this question.
//
// It does not for a boolean question. "yes" is not a span that appears in the
// supporting passage - the passage states two nationalities and the reader
// concludes they match - so looking for the gold answer in the text measures
// nothing about retrieval. Such questions are excluded from the metric rather
// than scored, the same way Recall@K is only computed where gold documents are
// annotated.
func AnswerInContextApplicable(golden []string) bool {
	for _, g := range golden {
		switch Normalize(g) {
		case "yes", "no":
			return false
		}
	}
	return len(golden) > 0
}

// containsSequence reports whether needle appears in haystack as a run of
// consecutive tokens.
func containsSequence(haystack, needle []string) bool {
	if len(needle) == 0 || len(needle) > len(haystack) {
		return false
	}

	for i := 0; i+len(needle) <= len(haystack); i++ {
		match := true
		for j, tok := range needle {
			if haystack[i+j] != tok {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
