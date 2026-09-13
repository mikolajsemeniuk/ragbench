package metrics

import "strings"

// DefaultAbstentionMaxWords is the length above which an answer is treated as
// a refusal rather than an answer. The gold answers in these datasets are
// short spans and the reader is instructed to output nothing else, so an
// answer of more than twenty words is not an attempt at the span.
const DefaultAbstentionMaxWords = 20

// IsAbstention reports whether the model declined to answer instead of
// producing a span.
//
// This exists because Exact Match cannot tell a refusal from a wrong answer,
// and the two are not the same thing at all. Measured on 2WikiMultihopQA:
// NaiveRAG produces an answer longer than twenty words on 16.8% of questions
// against ClosedBook's 2.3%, and every one of them scores Exact Match 0. The
// reason is that a reader holding irrelevant documents says "the provided
// documents do not contain information about ...", while a reader holding no
// documents at all has nothing to object to and guesses - and a guess is
// sometimes right (Exact Match 0.1889 on exactly those questions).
//
// Without this metric the resulting table says ClosedBook beats NaiveRAG on
// 2WikiMultihopQA (0.2444 against 0.2184). Restricted to the questions where
// NaiveRAG answered in under twenty words, the order reverses: 0.2625 against
// 0.2555. The gap measures willingness to abstain, not knowledge, and it has
// to be reported as such.
//
// Two signals, because either alone is leaky: an explicit refusal phrase, or
// an answer too long to be one of these datasets' spans. maxWords <= 0
// disables the length signal.
func IsAbstention(answer string, maxWords int) bool {
	normalised := Normalize(answer)
	for _, phrase := range abstentionPhrases {
		if strings.Contains(normalised, phrase) {
			return true
		}
	}
	return maxWords > 0 && len(strings.Fields(answer)) > maxWords
}

// abstentionPhrases are normalised at init with the same function the answers
// are, so that punctuation and articles cannot make a phrase miss.
var abstentionPhrases []string

func init() {
	for _, phrase := range []string{
		"do not contain",
		"does not contain",
		"no information",
		"not mentioned",
		"cannot answer",
		"can not answer",
		"not provided",
		"no mention",
		"unable to answer",
		"not enough information",
		"insufficient information",
		"there is no",
		"not possible to determine",
	} {
		abstentionPhrases = append(abstentionPhrases, Normalize(phrase))
	}
}
