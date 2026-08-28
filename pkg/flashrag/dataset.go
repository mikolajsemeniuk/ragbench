// Package flashrag holds the on-disk formats of the FlashRAG datasets and
// corpora. Every command that reads them goes through here, so the gold
// annotation is interpreted identically by the benchmark and by the analyses
// run on its output - a divergence there would silently make the two
// disagree.
package flashrag

import (
	"encoding/json"
	"strings"

	"golang.org/x/text/unicode/norm"
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

// PassageTitle pulls the article title out of a FlashRAG passage.
//
// The first line is the title, quoted for multi-word titles and bare for
// single-word ones - the corpus genuinely mixes both forms
// ("\"Evan Morris\"\ntext" but "Absalon\ntext"), so the quotes are trimmed
// rather than required.
func PassageTitle(contents string) string {
	line, _, _ := strings.Cut(contents, "\n")
	return strings.Trim(line, "\"")
}

// NormalizeTitle canonicalises an article title so that the corpus and the
// question sets can be compared.
//
// Comparing titles byte for byte counts an article that IS in the corpus as
// missing. Measured over the gold titles of the dev splits against the
// 3,232,908 titles of wiki18_100w, normalising recovers 4.5% of
// 2WikiMultihopQA's gold titles, 2.2% of HotpotQA's and 2.0% of MuSiQue's -
// which is a systematic understatement of Recall@K and MRR, and a
// corresponding overstatement of "gold article not in corpus" in the
// diagnosis.
//
// Almost all of it is Unicode composition rather than case: the corpus and the
// datasets disagree on whether an accented letter is stored precomposed
// (U+00E9) or as a base letter plus a combining mark (e + U+0301). Both look
// identical and neither is wrong. NFKC folds them together; lowercasing alone
// recovers 0.06% and would not be worth doing on its own.
func NormalizeTitle(title string) string {
	return strings.Join(strings.Fields(strings.ToLower(norm.NFKC.String(title))), " ")
}

// NormalizeTitles applies NormalizeTitle to a list, for the comparisons in
// cmd/bench where both sides have to be normalised the same way.
func NormalizeTitles(titles []string) []string {
	out := make([]string, len(titles))
	for i, t := range titles {
		out[i] = NormalizeTitle(t)
	}
	return out
}
