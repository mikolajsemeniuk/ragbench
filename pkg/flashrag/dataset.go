// Package flashrag holds the on-disk formats of the FlashRAG datasets and
// corpora. Every command that reads them goes through here, so the gold
// annotation is interpreted identically by the benchmark and by the analyses
// run on its output - a divergence there would silently make the two
// disagree.
package flashrag

import (
	"encoding/json"
	"strings"
)

// Question mirrors the FlashRAG question-set format. Metadata is kept raw
// because its shape differs between datasets - see GoldTitles.
type Question struct {
	Question      string          `json:"question"`
	GoldenAnswers []string        `json:"golden_answers"`
	Metadata      json.RawMessage `json:"metadata"`
}

// CorpusLine mirrors the FlashRAG corpus format (e.g. wiki18_100w.jsonl):
// {"id": "0", "contents": "\"Title\"\nPassage text..."}
type CorpusLine struct {
	ID       string `json:"id"`
	Contents string `json:"contents"`
}

// metadata covers the two shapes of gold-document annotation found in the
// FlashRAG datasets:
//   - HotpotQA / 2WikiMultihopQA: metadata.supporting_facts.title
//   - MuSiQue: metadata.question_decomposition[].support_paragraph.title
//
// NaturalQuestions and TriviaQA carry no such annotation.
type metadata struct {
	SupportingFacts struct {
		Title []string `json:"title"`
	} `json:"supporting_facts"`
	QuestionDecomposition []struct {
		SupportParagraph struct {
			Title string `json:"title"`
		} `json:"support_paragraph"`
	} `json:"question_decomposition"`
}

// GoldTitles returns the unique gold document titles for a question if the
// dataset annotates them, and nil otherwise.
func GoldTitles(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}

	var m metadata
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}

	seen := make(map[string]struct{})
	var titles []string
	add := func(t string) {
		if t == "" {
			return
		}
		if _, ok := seen[t]; ok {
			return
		}
		seen[t] = struct{}{}
		titles = append(titles, t)
	}

	for _, t := range m.SupportingFacts.Title {
		add(t)
	}
	for _, qd := range m.QuestionDecomposition {
		add(qd.SupportParagraph.Title)
	}
	return titles
}

// PassageTitle pulls the article title out of a FlashRAG passage, whose first
// line is the quoted title, e.g. "\"Title\"\nText...".
func PassageTitle(contents string) string {
	line, _, _ := strings.Cut(contents, "\n")
	return strings.Trim(line, "\"")
}
