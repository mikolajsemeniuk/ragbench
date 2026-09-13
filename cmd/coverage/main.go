// Command coverage reports the ceiling on Recall that the corpus imposes.
//
// A retrieval number is unreadable without it. NaiveRAG scores Recall 0.3008
// over its context on 2WikiMultihopQA, which looks like a weak retriever until
// one knows that only 67% of the articles the dataset annotates as gold exist
// in wiki18_100w at all - so a perfect retriever would score 0.67, and 0.3008
// is 45% of what is achievable rather than 30% of it.
//
// The gap has two very different causes and this command separates them.
// Titles that match only after normalisation are a measurement bug: the
// article is in the corpus and the comparison missed it. Titles that match
// under neither are a property of the data - the question sets were built
// against a different Wikipedia snapshot than the retrieval corpus - and are
// not fixable by any retriever.
//
// Run once per question set; the result belongs in the setup section of the
// paper, next to the Recall column.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/ragbench/pkg/flashrag"
)

func main() {
	var (
		datasetPath = flag.String("dataset", "", "FlashRAG question file to measure (required)")
		corpusPath  = flag.String("corpus", "dataset/wiki18_100w.jsonl", "FlashRAG corpus the collection was built from")
		texOut      = flag.String("tex-out", "", "path of the .tex file to write (e.g. paper/coverage-2wiki.gen.tex)")
		name        = flag.String("name", "Coverage", "name used in the generated .tex commands")
	)
	flag.Parse()

	if *datasetPath == "" {
		log.Fatal("missing required flag -dataset")
	}

	questions, err := loadGold(*datasetPath)
	if err != nil {
		log.Fatalf("reading the question set: %v", err)
	}
	if len(questions) == 0 {
		log.Fatalf("%s annotates no gold documents, so there is no coverage to measure (NaturalQuestions and TriviaQA are like this)", *datasetPath)
	}

	// Only membership is needed, so the corpus is reduced to two lookup sets
	// over the wanted titles instead of the 3.2M titles it actually holds.
	wantedExact := make(map[string]bool)
	wantedNorm := make(map[string]bool)
	for _, titles := range questions {
		for _, t := range titles {
			wantedExact[t] = false
			wantedNorm[flashrag.NormalizeTitle(t)] = false
		}
	}

	log.Printf("scanning %s for %d gold article titles", *corpusPath, len(wantedExact))
	start := time.Now()
	if err := scanCorpus(*corpusPath, wantedExact, wantedNorm); err != nil {
		log.Fatalf("scanning the corpus: %v", err)
	}
	log.Printf("scanned in %s", time.Since(start).Round(time.Second))

	var titlesExact, titlesNorm int
	for _, found := range wantedExact {
		if found {
			titlesExact++
		}
	}
	for _, found := range wantedNorm {
		if found {
			titlesNorm++
		}
	}

	var ceilingExact, ceilingNorm float64
	var fullExact, fullNorm int
	for _, titles := range questions {
		var ex, nz int
		for _, t := range titles {
			if wantedExact[t] {
				ex++
			}
			if wantedNorm[flashrag.NormalizeTitle(t)] {
				nz++
			}
		}
		ceilingExact += float64(ex) / float64(len(titles))
		ceilingNorm += float64(nz) / float64(len(titles))
		if ex == len(titles) {
			fullExact++
		}
		if nz == len(titles) {
			fullNorm++
		}
	}
	n := float64(len(questions))
	ceilingExact /= n
	ceilingNorm /= n

	fmt.Printf("--- corpus coverage: %s ---\n", *datasetPath)
	fmt.Printf("questions with gold annotation: %d\n", len(questions))
	fmt.Printf("unique gold article titles:     %d\n", len(wantedExact))
	fmt.Printf("  present, exact match:         %d (%.1f%%)\n", titlesExact, 100*float64(titlesExact)/float64(len(wantedExact)))
	fmt.Printf("  present, normalised match:    %d (%.1f%%)  <- the difference is a measurement bug, not missing data\n", titlesNorm, 100*float64(titlesNorm)/float64(len(wantedNorm)))
	fmt.Printf("ceiling on Recall (mean fraction of a question's gold articles that exist in the corpus)\n")
	fmt.Printf("  exact match:                  %.4f\n", ceilingExact)
	fmt.Printf("  normalised match:             %.4f  <- the number to read Recall against\n", ceilingNorm)
	fmt.Printf("questions whose gold articles are ALL in the corpus\n")
	fmt.Printf("  exact match:                  %d (%.1f%%)\n", fullExact, 100*float64(fullExact)/n)
	fmt.Printf("  normalised match:             %d (%.1f%%)\n", fullNorm, 100*float64(fullNorm)/n)

	if *texOut != "" {
		if err := writeTex(*texOut, *name, len(questions), len(wantedExact), titlesExact, titlesNorm, ceilingExact, ceilingNorm, fullExact, fullNorm); err != nil {
			log.Fatalf("writing the tex file: %v", err)
		}
		log.Printf("written to %s", *texOut)
	}
}

// scanCorpus marks every wanted title that occurs in the corpus, in one pass.
func scanCorpus(path string, wantedExact, wantedNorm map[string]bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

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
		title := flashrag.PassageTitle(cl.Contents)
		if _, ok := wantedExact[title]; ok {
			wantedExact[title] = true
		}
		normalised := flashrag.NormalizeTitle(title)
		if _, ok := wantedNorm[normalised]; ok {
			wantedNorm[normalised] = true
		}
	}
	return scanner.Err()
}

// loadGold returns the gold article titles of every question that has any.
func loadGold(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out [][]string
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
		if titles := flashrag.GoldTitles(q.Metadata); len(titles) > 0 {
			out = append(out, titles)
		}
	}
	return out, scanner.Err()
}

func writeTex(path, name string, questions, titles, presentExact, presentNorm int, ceilingExact, ceilingNorm float64, fullExact, fullNorm int) error {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	id := texSafeID(name)
	var b strings.Builder
	fmt.Fprintf(&b, "%% generated automatically by cmd/coverage - do not edit by hand\n")
	fmt.Fprintf(&b, "\\newcommand{\\%sAnnotatedQuestions}{%d}\n", id, questions)
	fmt.Fprintf(&b, "\\newcommand{\\%sGoldTitles}{%d}\n", id, titles)
	fmt.Fprintf(&b, "\\newcommand{\\%sGoldTitlesPresent}{%d}\n", id, presentNorm)
	fmt.Fprintf(&b, "\\newcommand{\\%sGoldTitlesPresentExact}{%d}\n", id, presentExact)
	fmt.Fprintf(&b, "\\newcommand{\\%sRecallCeiling}{%.4f}\n", id, ceilingNorm)
	fmt.Fprintf(&b, "\\newcommand{\\%sRecallCeilingExact}{%.4f}\n", id, ceilingExact)
	fmt.Fprintf(&b, "\\newcommand{\\%sQuestionsFullyCovered}{%d}\n", id, fullNorm)
	fmt.Fprintf(&b, "\\newcommand{\\%sQuestionsFullyCoveredPct}{%.1f}\n", id, 100*float64(fullNorm)/float64(questions))
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// texSafeID drops every character outside [A-Za-z], because a LaTeX
// \newcommand name may contain letters only.
func texSafeID(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		}
	}
	return b.String()
}
