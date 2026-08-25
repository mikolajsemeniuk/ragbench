package metrics

import "strings"

// AnswerInContext returns 1 if any retrieved passage contains one of the gold
// answers, and 0 otherwise.
//
// This is the retrieval metric for datasets that annotate no gold documents at
// all - NaturalQuestions and TriviaQA are open-domain question sets with no
// supporting-passage labels, so title-based Recall@K cannot be computed for
// them. It answers the question that actually matters for a RAG pipeline: did
// retrieval put the information the generator needs in front of it?
//
// It is a proxy, not ground truth: a passage can contain the answer string
// incidentally, without supporting the question. Reported alongside Recall@K
// rather than instead of it.
func AnswerInContext(passages []string, golden []string) float64 {
	normalised := make([]string, len(passages))
	for i, p := range passages {
		normalised[i] = Normalize(p)
	}

	for _, g := range golden {
		ng := Normalize(g)
		if ng == "" {
			continue
		}
		for _, p := range normalised {
			if strings.Contains(p, ng) {
				return 1
			}
		}
	}
	return 0
}
