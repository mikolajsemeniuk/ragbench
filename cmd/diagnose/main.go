// Command diagnose explains *why* retrieval failed, which the benchmark
// numbers alone cannot say.
//
// cmd/bench reports that a gold answer was absent from the retrieved context
// for, say, 66% of 2WikiMultihopQA questions. That single number hides several
// causes that call for completely different fixes:
//
//   - the right article was retrieved, but the wrong ~100-word slice of it;
//   - the right article exists and is reachable, but ranked too low to be cut;
//   - the right article is unreachable by this query, because the question
//     never names the entity that would find it;
//   - the article is not in the corpus at all.
//
// Only the first three are addressable, and each by different means, so the
// split determines what is worth building next. It is written to a .tex
// fragment alongside the benchmark results.
//
// Given -closedbook it also reports the two comparisons that motivate a
// retrieve-or-abstain design: how the system does when retrieval succeeds
// versus when it fails, and the ceiling of always picking the better of the
// two systems per question.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/ragbench/pkg/eval"
	"github.com/mikolajsemeniuk/ragbench/pkg/flashrag"
	"github.com/mikolajsemeniuk/ragbench/pkg/metrics"
	"github.com/mikolajsemeniuk/ragbench/pkg/provider"
	"github.com/mikolajsemeniuk/ragbench/pkg/storage"
)

// runRecord is one line of a cmd/bench -dump file.
type runRecord struct {
	Question string   `json:"question"`
	Gold     []string `json:"gold"`
	EM       float64  `json:"em"`

	// InCtx is only meaningful where InCtxApplicable is set: a boolean
	// question has no answer span in the supporting passage, so scoring one
	// would send every yes/no question into the diagnosis as a retrieval
	// failure it is not.
	InCtx           float64 `json:"answer_in_context"`
	InCtxApplicable bool    `json:"answer_in_context_applicable"`

	Abstained bool `json:"abstained"`
}

type passage struct {
	id   uint64
	text string
}

func main() {
	var (
		datasetPath = flag.String("dataset", "", "FlashRAG question file the run was evaluated on (required)")
		corpusPath  = flag.String("corpus", "dataset/wiki18_100w.jsonl", "FlashRAG corpus the collection was built from")
		runPath     = flag.String("run", "", "cmd/bench -dump file of the run to diagnose (required)")
		cbPath      = flag.String("closedbook", "", "cmd/bench -dump file of a closed-book run on the same dataset - enables the retrieval-succeeded/failed split and the oracle ceiling")

		embedURL   = flag.String("embed-url", "http://localhost:8001", "embedding provider URL")
		embedModel = flag.String("embed-model", "bge-base-en-v1.5", "embedding model name")
		qdrantURL  = flag.String("qdrant-url", "http://localhost:6333", "Qdrant URL")
		collection = flag.String("collection", "ragbench", "Qdrant collection the run used")

		sample = flag.Int("sample", 300, "how many failed questions to diagnose; 0 = all of them. Each one costs a deep search, so a sample keeps the tool quick while the reported counts stay in the .tex")
		seed   = flag.Int64("seed", 42, "seed for the -sample draw. The sample is drawn at random rather than taken from the front of the file: these datasets are not shuffled and 2WikiMultihopQA in particular groups questions by type, so the first N failures can describe one question type only")
		hnswEf = flag.Int("hnsw-ef", 1024, "size of the candidate list Qdrant keeps while walking the HNSW graph. It has to be at least -deep, or a gold article counted as \"not reachable by this query\" may only have been missed by the graph walk")
		deep   = flag.Int("deep", 1000, "how far down the ranking to look for the gold article. Anything below this is reported as unreachable by this query")
		topK   = flag.Int("top-k", 10, "ranking cut treated as \"the retriever did surface the article\"")

		jsonOut = flag.String("json-out", "", "path of the eval .json file to write (e.g. eval/diagnosis-2wiki.json); rendered to LaTeX by cmd/render")
		name    = flag.String("name", "Diagnosis", "prefix of the aggregate names")
	)
	flag.Parse()

	if *datasetPath == "" || *runPath == "" {
		log.Fatal("-dataset and -run are both required")
	}

	ctx := context.Background()
	start := time.Now()

	goldTitles, err := loadGoldTitles(*datasetPath)
	if err != nil {
		log.Fatalf("reading dataset: %v", err)
	}
	run, err := loadRun(*runPath)
	if err != nil {
		log.Fatalf("reading run dump: %v", err)
	}

	// The questions to diagnose: retrieval did not put a gold answer in
	// context, and the dataset annotates which articles it should have found.
	var failed []runRecord
	for _, r := range run {
		if r.InCtxApplicable && r.InCtx == 0 && len(goldTitles[r.Question]) > 0 {
			failed = append(failed, r)
		}
	}
	if len(failed) == 0 {
		log.Fatal("no failed questions with gold annotations in this run - nothing to diagnose")
	}
	diagnosed := failed
	if *sample > 0 && *sample < len(failed) {
		shuffled := append([]runRecord(nil), failed...)
		rng := rand.New(rand.NewSource(*seed))
		rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
		diagnosed = shuffled[:*sample]
		log.Printf("diagnosing a random sample of %d of them (seed=%d)", *sample, *seed)
	}
	log.Printf("retrieval failed on %d of %d annotated questions; diagnosing %d of them", len(failed), len(run), len(diagnosed))

	// Titles are normalised on both sides, the same way cmd/bench scores
	// Recall: the corpus and the datasets disagree on Unicode composition
	// often enough that a byte comparison reports articles as absent from the
	// corpus when they are in it.
	wanted := make(map[string]struct{})
	for _, r := range diagnosed {
		for _, t := range goldTitles[r.Question] {
			wanted[flashrag.NormalizeTitle(t)] = struct{}{}
		}
	}

	log.Printf("scanning %s for %d gold article titles", *corpusPath, len(wanted))
	byTitle, err := scanCorpus(*corpusPath, wanted)
	if err != nil {
		log.Fatalf("scanning corpus: %v", err)
	}
	log.Printf("found %d of them in the corpus (%s)", len(byTitle), time.Since(start).Round(time.Second))

	embedder := provider.NewVLLM(*embedURL, *embedModel)
	embedder.TruncatePromptTokens = 512
	store := storage.NewQdrant(*qdrantURL)
	store.HNSWEf = *hnswEf
	if *hnswEf > 0 && *hnswEf < *deep {
		log.Printf("WARNING: -hnsw-ef %d is below -deep %d, so an article may be reported unreachable when the graph walk simply did not visit it", *hnswEf, *deep)
	}

	var (
		notInCorpus, rankedLow, unreachable int
		wrongSlice, answerAbsent            int
	)
	for i := 0; i < len(diagnosed); i += 64 {
		end := min(i+64, len(diagnosed))
		batch := diagnosed[i:end]

		texts := make([]string, len(batch))
		for j, r := range batch {
			texts[j] = r.Question
		}
		vectors, err := embedder.Embed(ctx, texts)
		if err != nil {
			log.Fatalf("embedding questions: %v", err)
		}

		for j, r := range batch {
			var golds []passage
			for _, t := range flashrag.NormalizeTitles(goldTitles[r.Question]) {
				golds = append(golds, byTitle[t]...)
			}
			if len(golds) == 0 {
				notInCorpus++
				continue
			}

			// Classify against the article that actually holds the answer, not
			// against "any gold article". A two-hop question annotates two gold
			// articles; retrieving the one that does NOT contain the answer is
			// not a chunking problem, and expanding it with its neighbouring
			// passages cannot help. Conflating the two inflates the fixable
			// bucket and points at the wrong remedy.
			holders := holderIDs(r.Gold, golds)
			if len(holders) == 0 {
				answerAbsent++
				continue
			}

			hits, err := store.Search(ctx, *collection, vectors[j], *deep)
			if err != nil {
				log.Fatalf("deep search: %v", err)
			}

			// Where does the answer-bearing article first appear? Any of its
			// passages counts, because neighbour expansion pulls in the rest
			// once one is retrieved.
			article := holderArticles(holders, byTitle, flashrag.NormalizeTitles(goldTitles[r.Question]))
			rank := 0
			for k, h := range hits {
				if _, ok := article[h.ID]; ok {
					rank = k + 1
					break
				}
			}

			switch {
			case rank > 0 && rank <= *topK:
				wrongSlice++
			case rank > 0:
				rankedLow++
			default:
				unreachable++
			}
		}
		log.Printf("  %d/%d diagnosed", end, len(diagnosed))
	}

	n := float64(len(diagnosed))
	pct := func(c int) float64 { return 100 * float64(c) / n }

	fmt.Printf("\n--- why retrieval failed: %s ---\n", *runPath)
	fmt.Printf("questions in run:            %d\n", len(run))
	fmt.Printf("retrieval failed on:         %d (%.1f%% of the run)\n", len(failed), 100*float64(len(failed))/float64(len(run)))
	fmt.Printf("diagnosed:                   %d\n\n", len(diagnosed))
	fmt.Printf("answer-bearing article in top-%-2d %5d  (%4.1f%%)  wrong slice of it: pull neighbouring passages\n", *topK, wrongSlice, pct(wrongSlice))
	fmt.Printf("...ranked %d-%-5d               %5d  (%4.1f%%)  fix: better ranking / hybrid search\n", *topK+1, *deep, rankedLow, pct(rankedLow))
	fmt.Printf("...not found within %-5d        %5d  (%4.1f%%)  fix: decompose the question\n", *deep, unreachable, pct(unreachable))
	fmt.Printf("no gold article states the answer%4d  (%4.1f%%)  not a retrieval failure: needs inference\n", answerAbsent, pct(answerAbsent))
	fmt.Printf("gold article not in corpus      %5d  (%4.1f%%)  not fixable\n", notInCorpus, pct(notInCorpus))

	var extra []eval.Command
	if *cbPath != "" {
		extra = compareToClosedBook(*cbPath, run)
	}

	if *jsonOut != "" {
		if err := writeJSON(*jsonOut, *name, len(run), len(failed), len(diagnosed),
			wrongSlice, answerAbsent, rankedLow, unreachable, notInCorpus, *topK, *deep, extra); err != nil {
			log.Fatalf("writing eval file: %v", err)
		}
		log.Printf("written to %s", *jsonOut)
	}
	log.Printf("done in %s", time.Since(start).Round(time.Second))
}

// holderIDs returns the passages that actually state a gold answer, using the
// same normalisation the benchmark scores with.
func holderIDs(gold []string, passages []passage) map[uint64]struct{} {
	out := make(map[uint64]struct{})
	for _, p := range passages {
		text := metrics.Normalize(p.text)
		for _, g := range gold {
			ng := metrics.Normalize(g)
			if ng != "" && strings.Contains(text, ng) {
				out[p.id] = struct{}{}
				break
			}
		}
	}
	return out
}

// holderArticles returns every passage belonging to an article that holds a
// gold answer - the set neighbour expansion would pull in once any one of them
// is retrieved.
func holderArticles(holders map[uint64]struct{}, byTitle map[string][]passage, titles []string) map[uint64]struct{} {
	holderTitles := make(map[string]struct{})
	for _, t := range titles {
		for _, p := range byTitle[t] {
			if _, ok := holders[p.id]; ok {
				holderTitles[t] = struct{}{}
				break
			}
		}
	}

	out := make(map[uint64]struct{})
	for t := range holderTitles {
		for _, p := range byTitle[t] {
			out[p.id] = struct{}{}
		}
	}
	return out
}

// compareToClosedBook prints, and returns as aggregates (the LaTeX macro
// names are prefixed with the run's -name by writeJSON), the two figures that
// argue for a retrieve-or-abstain design: how the system does when retrieval
// succeeded versus when it failed, and the ceiling of picking the better of
// the two systems per question.
func compareToClosedBook(path string, run []runRecord) []eval.Command {
	cb, err := loadRun(path)
	if err != nil {
		log.Fatalf("reading closed-book dump: %v", err)
	}
	byQuestion := make(map[string]runRecord, len(cb))
	for _, r := range cb {
		byQuestion[r.Question] = r
	}

	var foundN, missedN int
	var foundRAG, foundCB, missedRAG, missedCB, oracleSum, ragSum, cbSum float64
	var foundAbstain, missedAbstain, cbAbstain float64
	var paired int
	for _, r := range run {
		c, ok := byQuestion[r.Question]
		if !ok {
			continue
		}
		paired++
		ragSum += r.EM
		cbSum += c.EM
		cbAbstain += boolean(c.Abstained)
		oracleSum += max(r.EM, c.EM)
		if r.InCtx == 1 {
			foundN++
			foundRAG += r.EM
			foundCB += c.EM
			foundAbstain += boolean(r.Abstained)
		} else {
			missedN++
			missedRAG += r.EM
			missedCB += c.EM
			missedAbstain += boolean(r.Abstained)
		}
	}
	if paired == 0 {
		log.Printf("closed-book dump shares no questions with the run - skipping that analysis")
		return nil
	}

	fmt.Printf("\n--- retrieval succeeded vs failed (paired with %s) ---\n", path)
	fmt.Printf("%-22s %8s %10s %12s %10s %12s\n", "", "n", "this run", "closed-book", "diff", "abstained")
	fr, fc := foundRAG/float64(foundN), foundCB/float64(foundN)
	mr, mc := missedRAG/float64(missedN), missedCB/float64(missedN)
	fa, ma := foundAbstain/float64(foundN), missedAbstain/float64(missedN)
	fmt.Printf("%-22s %8d %10.4f %12.4f %+10.4f %12.4f\n", "answer was retrieved", foundN, fr, fc, fr-fc, fa)
	fmt.Printf("%-22s %8d %10.4f %12.4f %+10.4f %12.4f\n", "answer was not", missedN, mr, mc, mr-mc, ma)
	// The abstention columns are what stops the closed-book column from being
	// misread. When retrieval fails, this run declines and scores Exact Match
	// 0 while the closed-book system, having nothing to object to, guesses -
	// and a guess is sometimes right. That is a difference in behaviour, not
	// in knowledge.
	fmt.Printf("%-22s %8d %10s %12.4f\n", "closed-book abstained", paired, "-", cbAbstain/float64(paired))

	r, c, o := ragSum/float64(paired), cbSum/float64(paired), oracleSum/float64(paired)
	fmt.Printf("\n--- ceiling of choosing per question ---\n")
	fmt.Printf("this run %.4f | closed-book %.4f | always the better one %.4f (+%.4f over the best single system)\n",
		r, c, o, o-max(r, c))

	return []eval.Command{
		{Name: "RetrievedN", Value: fmt.Sprintf("%d", foundN)},
		{Name: "RetrievedEM", Value: fmt.Sprintf("%.4f", fr)},
		{Name: "RetrievedClosedBookEM", Value: fmt.Sprintf("%.4f", fc)},
		{Name: "RetrievedAbstention", Value: fmt.Sprintf("%.4f", fa)},
		{Name: "NotRetrievedN", Value: fmt.Sprintf("%d", missedN)},
		{Name: "NotRetrievedEM", Value: fmt.Sprintf("%.4f", mr)},
		{Name: "NotRetrievedClosedBookEM", Value: fmt.Sprintf("%.4f", mc)},
		{Name: "NotRetrievedAbstention", Value: fmt.Sprintf("%.4f", ma)},
		{Name: "ClosedBookAbstention", Value: fmt.Sprintf("%.4f", cbAbstain/float64(paired))},
		{Name: "ClosedBookEM", Value: fmt.Sprintf("%.4f", c)},
		{Name: "OracleEM", Value: fmt.Sprintf("%.4f", o)},
		{Name: "OracleGain", Value: fmt.Sprintf("%.4f", o-max(r, c))},
	}
}

func boolean(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

// scanCorpus returns the passages of every wanted article title.
//
// The corpus is parsed as JSON rather than scanned for raw byte patterns:
// non-ASCII characters are stored escaped (ł for ł), so matching a title
// like "Małgorzata Braunek" against the raw bytes never fires and silently
// inflates the "not in corpus" count.
func scanCorpus(path string, wanted map[string]struct{}) (map[string][]passage, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	out := make(map[string][]passage, len(wanted))
	scanner := bufio.NewScanner(bufio.NewReaderSize(f, 8*1024*1024))
	scanner.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var cl flashrag.CorpusLine
		if err := json.Unmarshal(line, &cl); err != nil {
			continue
		}
		title := flashrag.NormalizeTitle(flashrag.PassageTitle(cl.Contents))
		if _, ok := wanted[title]; !ok {
			continue
		}
		id, err := strconv.ParseUint(cl.ID, 10, 64)
		if err != nil {
			continue
		}
		out[title] = append(out[title], passage{id: id, text: cl.Contents})
	}
	return out, scanner.Err()
}

func loadGoldTitles(path string) (map[string][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	out := make(map[string][]string)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var q flashrag.Question
		if err := json.Unmarshal([]byte(line), &q); err != nil {
			return nil, err
		}
		out[q.Question] = flashrag.GoldTitles(q.Metadata)
	}
	return out, scanner.Err()
}

// checkSchema refuses a dump written before the metrics were fixed. Without
// "answer_in_context_applicable" every record would look inapplicable, the
// failure list would come out empty, and the tool would report that there is
// nothing to diagnose rather than that the file is stale.
func checkSchema(line []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(line, &fields); err != nil {
		return fmt.Errorf("parse first record: %w", err)
	}
	for _, required := range []string{"abstained", "answer_in_context_applicable"} {
		if _, ok := fields[required]; !ok {
			return fmt.Errorf("this dump predates the current metrics (no %q field) - re-run cmd/bench to regenerate it", required)
		}
	}
	return nil
}

func loadRun(path string) ([]runRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []runRecord
	checked := false
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		if !checked {
			if err := checkSchema(line); err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			checked = true
		}
		var r runRecord
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, scanner.Err()
}

func writeJSON(path, name string, runN, failedN, diagnosedN, wrongSlice, answerAbsent, rankedLow, unreachable, notInCorpus, topK, deep int, extra []eval.Command) error {
	id := texSafeID(name)
	pct := func(c int) float64 { return 100 * float64(c) / float64(diagnosedN) }

	doc := eval.Doc{Generator: "cmd/diagnose"}
	doc.Addf(id+"Questions", "%d", runN)
	doc.Addf(id+"Failed", "%d", failedN)
	doc.Addf(id+"FailedPct", "%.1f", 100*float64(failedN)/float64(runN))
	doc.Addf(id+"Diagnosed", "%d", diagnosedN)
	doc.Addf(id+"TopK", "%d", topK)
	doc.Addf(id+"Deep", "%d", deep)
	for _, row := range []struct {
		key   string
		count int
	}{
		{"WrongSlice", wrongSlice},
		{"AnswerAbsent", answerAbsent},
		{"RankedLow", rankedLow},
		{"Unreachable", unreachable},
		{"NotInCorpus", notInCorpus},
	} {
		doc.Addf(id+row.key, "%d", row.count)
		doc.Addf(id+row.key+"Pct", "%.1f", pct(row.count))
	}
	for _, c := range extra {
		doc.Add(id+c.Name, c.Value)
	}
	return eval.Write(path, &doc)
}

func texSafeID(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		}
	}
	return b.String()
}
