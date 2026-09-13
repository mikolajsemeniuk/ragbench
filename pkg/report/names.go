// Package report holds what the table-rendering commands (cmd/result,
// cmd/cost) share: the display names of the architectures and question sets,
// so that a row is labelled the same way in every generated table.
package report

// ArchitectureName maps a run slug (the prefix of runs/<slug>-<set>.jsonl) to
// the label used in the paper. An unknown slug is shown as is.
func ArchitectureName(slug string) string {
	if name, ok := architectureNames[slug]; ok {
		return name
	}
	return slug
}

// SetName maps a question-set slug to its label.
func SetName(slug string) string {
	if name, ok := setNames[slug]; ok {
		return name
	}
	return slug
}

var architectureNames = map[string]string{
	"closedbook":     "ClosedBook (no retrieval)",
	"bm25":           "BM25",
	"naive":          "NaiveRAG, 5 passages",
	"naive10":        "NaiveRAG, 10 passages",
	"naive12":        "NaiveRAG, 12 passages",
	"hybrid":         "Hybrid (dense + BM25)",
	"hyde":           "HyDE",
	"rerank":         "Rerank (cross-encoder)",
	"neighbour":      "Neighbour expansion",
	"crag":           "CRAG (offline), 5 passages",
	"crag10":         "CRAG (offline), 10 passages",
	"ircot":          "IRCoT",
	"ircot1":         "IRCoT, one-shot",
	"adaptive":       "Adaptive-RAG",
	"fused":          "Fused retrieval (stage 1 alone)",
	"cascade":        "Cascade",
	"cascade-lp":     "Cascade + confidence",
	"cascade-rr":     "Cascade, rerank first",
	"cascade-naive":  "Cascade, naive first",
	"cascade-hybrid": "Cascade, hybrid first",
}

var setNames = map[string]string{
	"nq":       "NQ",
	"triviaqa": "TriviaQA",
	"hotpotqa": "HotpotQA",
	"2wiki":    "2Wiki",
	"musique":  "MuSiQue",
}
