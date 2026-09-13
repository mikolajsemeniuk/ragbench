package rag

import (
	"hash/fnv"
	"strings"
	"unicode"

	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// BM25 turns text into the sparse vectors a lexical index is built from.
//
// Why a lexical index at all, next to a perfectly good dense one: a
// bi-encoder compresses a passage into 768 numbers, and rare surface forms are
// exactly what that compression loses. A question like "Which film has the
// director died earlier, Poker In Bed or The Machine To Kill Bad People?"
// names its target articles verbatim, and the dense retriever still has to
// hope those names survived the embedding. An inverted index does not hope -
// it looks the term up. The measured failure bucket this addresses is
// "unreachable by this query" (cmd/diagnose): 24.7% of failures on
// 2WikiMultihopQA, 23.0% on HotpotQA, 53.3% on MuSiQue.
//
// The scoring is standard BM25 split across client and server. The client
// stores, per term, the term-frequency saturation factor
//
//	tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
//
// and the query vector carries a plain 1.0 per term. Qdrant's "idf" modifier
// supplies the inverse document frequency at query time. Multiplying the two
// is BM25, and it means the client never has to make a pass over the corpus to
// count document frequencies - which on 21M passages would be a second full
// read of a 14 GB file.
type BM25 struct {
	// K1 controls how quickly repeated occurrences of a term stop adding
	// score; B how strongly a long passage is penalised. 1.2 and 0.75 are the
	// values BM25 is almost always reported with.
	K1 float64
	B  float64

	// AvgDocLen is the mean number of indexed tokens per passage. Measured
	// over the FlashRAG wiki18_100w corpus: 98.2, which is what a "100 word"
	// chunking produces once punctuation and one-character tokens are
	// dropped.
	AvgDocLen float64
}

// DefaultAvgDocLen is the measured mean token count of a wiki18_100w passage.
const DefaultAvgDocLen = 98.0

func NewBM25() *BM25 {
	return &BM25{K1: 1.2, B: 0.75, AvgDocLen: DefaultAvgDocLen}
}

// Document encodes a passage for storage.
//
// The whole passage is used, title line included. The title is often the most
// discriminative text a passage has - it is the entity the question names -
// and the dense side embeds it too, so including it keeps the two retrievers
// looking at the same input.
func (b *BM25) Document(text string) storage.SparseVector {
	counts, length := termCounts(text)
	if len(counts) == 0 {
		return storage.SparseVector{}
	}

	norm := b.K1 * (1 - b.B + b.B*float64(length)/b.AvgDocLen)
	indices := make([]uint32, 0, len(counts))
	values := make([]float32, 0, len(counts))
	for term, tf := range counts {
		w := float64(tf) * (b.K1 + 1) / (float64(tf) + norm)
		indices = append(indices, term)
		values = append(values, float32(w))
	}
	return storage.SparseVector{Indices: indices, Values: values}
}

// Query encodes a question. Every term weighs the same: the differences
// between them are already expressed by the inverse document frequency Qdrant
// applies, and weighting a term twice would double-count it.
func (b *BM25) Query(text string) storage.SparseVector {
	counts, _ := termCounts(text)
	if len(counts) == 0 {
		return storage.SparseVector{}
	}

	indices := make([]uint32, 0, len(counts))
	values := make([]float32, 0, len(counts))
	for term := range counts {
		indices = append(indices, term)
		values = append(values, 1)
	}
	return storage.SparseVector{Indices: indices, Values: values}
}

// termCounts tokenises text and returns the frequency of each term id
// together with the total number of indexed tokens (the document length BM25
// normalises by).
func termCounts(text string) (map[uint32]int, int) {
	counts := make(map[uint32]int)
	length := 0
	for _, token := range tokenize(text) {
		counts[termID(token)]++
		length++
	}
	return counts, length
}

// tokenize lowercases, splits on anything that is not a letter or a digit, and
// drops one-character tokens and stop words.
//
// No stemming: it would need a dependency, and on a corpus this size the
// inverse document frequency already does most of the work a stemmer would.
// Stop words are dropped despite that, for size rather than quality - a
// posting list for "the" would hold roughly 20M of the 21M passages.
func tokenize(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	out := fields[:0]
	for _, f := range fields {
		if len(f) < 2 {
			continue
		}
		if _, stop := stopWords[f]; stop {
			continue
		}
		out = append(out, f)
	}
	return out
}

// termID hashes a token into the 32-bit space Qdrant's sparse index uses.
//
// Hashing rather than keeping a vocabulary avoids having to build, store and
// ship a term-to-id table for a corpus with millions of distinct tokens. The
// cost is collisions: with ~5M distinct terms in a 32-bit space roughly 3000
// pairs of terms are expected to share an id, i.e. about one term in 1600
// occasionally matches a passage it should not. That is far below the noise
// floor of the metrics reported here.
func termID(token string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(token))
	return h.Sum32()
}

var stopWords = map[string]struct{}{}

func init() {
	for _, w := range strings.Fields(`
		about after all also an and any are as at be because been but by can
		could did do does for from had has have he her hers him his how if in
		into is it its me more most my no nor not of off on once only or other
		our out over own same she should so some such than that the their them
		then there these they this those through to too under until up very
		was we were what when where which while who whom why will with would
		you your
	`) {
		stopWords[w] = struct{}{}
	}
}
