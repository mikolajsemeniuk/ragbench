package rag

import "strings"

// Some embedding models are trained asymmetrically: the query and the passage
// go through the same encoder, but the query is expected to carry a short
// natural-language instruction that the passage does not. Retrieval quality
// degrades when the convention is ignored, and it degrades in a way that is
// invisible from the outside - the index looks fine, the searches return
// results, they are just worse.
//
// The convention is per model family, so it lives in one table rather than
// being hardcoded at the call sites:
//
//   - BGE English v1.5: nothing on either side. The instruction is optional
//     for the v1.5 line, and measured on this benchmark it changes nothing:
//     over 500 HotpotQA dev questions against a 415k-passage index, the
//     paired 95% bootstrap interval for every retrieval metric straddles zero
//     (Recall@5 -0.0007 [-0.0055, +0.0041], MRR -0.0051 [-0.0152, +0.0038],
//     title coverage@5 +0.0040 [-0.0070, +0.0150]). Pass -query-prefix
//     explicitly to run it as an ablation.
//   - BGE English before v1.5: instruction on the query, nothing on the
//     passage - those checkpoints do depend on it.
//   - E5: "query: " / "passage: " - both sides prefixed, asymmetrically.
//   - Nomic Embed: "search_query: " / "search_document: ", likewise.
//   - BGE-M3, GTE and anything unrecognised: no prefixes.
//
// Note the asymmetry in what it costs to change these. QueryPrefix only
// affects the embedding of the question, so it can be changed and the
// benchmark re-run against an existing collection. DocumentPrefix is baked
// into the stored vectors, so changing it invalidates the collection and
// requires a full re-ingest.
const (
	// BGEEnglishQueryInstruction is the query instruction documented for the
	// English BGE retrieval models (FlagEmbedding's
	// query_instruction_for_retrieval).
	BGEEnglishQueryInstruction = "Represent this sentence for searching relevant passages: "

	E5QueryPrefix    = "query: "
	E5PassagePrefix  = "passage: "
	NomicQueryPrefix = "search_query: "
	NomicDocPrefix   = "search_document: "
)

// PrefixesFor returns the query and document prefixes conventionally used with
// the named embedding model. known reports whether the model was recognised;
// an unrecognised model gets no prefixes, which is the safe default.
func PrefixesFor(model string) (query, document string, known bool) {
	m := strings.ToLower(model)

	switch {
	// BGE-M3 is trained without instructions, unlike the English v1.5 line.
	case strings.Contains(m, "bge-m3"):
		return "", "", true
	// The v1.5 checkpoints were trained to work without the instruction; the
	// earlier ones were not.
	case strings.Contains(m, "bge") && strings.Contains(m, "-en") && strings.Contains(m, "v1.5"):
		return "", "", true
	case strings.Contains(m, "bge") && strings.Contains(m, "-en"):
		return BGEEnglishQueryInstruction, "", true
	case strings.Contains(m, "nomic-embed"):
		return NomicQueryPrefix, NomicDocPrefix, true
	case hasSegment(m, "e5"):
		return E5QueryPrefix, E5PassagePrefix, true
	}
	return "", "", false
}

// hasSegment reports whether name contains segment as a whole hyphen-, slash-
// or underscore-delimited part, so that "multilingual-e5-large" matches "e5"
// while an unrelated name that merely contains those two letters does not.
func hasSegment(name, segment string) bool {
	for _, part := range strings.FieldsFunc(name, func(r rune) bool {
		return r == '-' || r == '/' || r == '_' || r == '.'
	}) {
		if part == segment {
			return true
		}
	}
	return false
}

// ResolvePrefix turns a command-line prefix flag into the prefix to use.
// The sentinel "auto" selects the convention for the model; any other value,
// including the empty string, is taken literally - which is what makes an
// explicit "no instruction" ablation possible.
func ResolvePrefix(flagValue, model string, wantQuery bool) string {
	if flagValue != PrefixAuto {
		return flagValue
	}
	query, document, _ := PrefixesFor(model)
	if wantQuery {
		return query
	}
	return document
}

// PrefixAuto is the flag value that selects the model's documented
// convention.
const PrefixAuto = "auto"
