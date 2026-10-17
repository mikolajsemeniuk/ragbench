package flashrag

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strconv"
)

// TitleIndex maps an article title to the ids of every passage that belongs to
// it, in ascending order.
//
// It exists because the corpus is not laid out the way one would assume.
// wiki18_100w is not sorted by article: passages of the same article are near
// each other but interleaved with others, e.g. ids 29, 31 and 32 are
// "Absalon" while 30 is "Pseudorandom generator". Reaching the passage next to
// a retrieved one therefore cannot be done with id arithmetic; the article's
// own id list has to be known.
//
// Building it costs one pass over the 14 GB corpus, so the result is written
// to disk and reused - see OpenTitleIndex.
type TitleIndex struct {
	ByTitle map[string][]uint64
}

// BuildTitleIndex scans the corpus and groups passage ids by article title.
func BuildTitleIndex(corpusPath string) (*TitleIndex, error) {
	f, err := os.Open(corpusPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	byTitle := make(map[string][]uint64)
	scanner := bufio.NewScanner(bufio.NewReaderSize(f, 8*1024*1024))
	scanner.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var cl CorpusLine
		if err := json.Unmarshal(line, &cl); err != nil {
			continue
		}
		id, err := strconv.ParseUint(cl.ID, 10, 64)
		if err != nil {
			continue
		}
		title := PassageTitle(cl.Contents)
		byTitle[title] = append(byTitle[title], id)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// The corpus is read in file order, which is not id order for a given
	// article, so each list is sorted before it can be searched.
	for _, ids := range byTitle {
		slices.Sort(ids)
	}
	return &TitleIndex{ByTitle: byTitle}, nil
}

// OpenTitleIndex loads the index from indexPath, building it from corpusPath
// and saving it there when the file does not exist yet.
func OpenTitleIndex(indexPath, corpusPath string) (*TitleIndex, error) {
	ix, err := LoadTitleIndex(indexPath)
	if err == nil {
		return ix, nil
	}
	if !os.IsNotExist(err) {
		return nil, fmt.Errorf("load title index %s: %w", indexPath, err)
	}

	ix, err = BuildTitleIndex(corpusPath)
	if err != nil {
		return nil, fmt.Errorf("build title index from %s: %w", corpusPath, err)
	}
	if err := ix.Save(indexPath); err != nil {
		return nil, fmt.Errorf("save title index %s: %w", indexPath, err)
	}
	return ix, nil
}

func LoadTitleIndex(path string) (*TitleIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var ix TitleIndex
	if err := gob.NewDecoder(bufio.NewReaderSize(f, 8*1024*1024)).Decode(&ix); err != nil {
		return nil, err
	}
	return &ix, nil
}

func (ix *TitleIndex) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 8*1024*1024)
	if err := gob.NewEncoder(w).Encode(ix); err != nil {
		return err
	}
	return w.Flush()
}

// Passages returns the number of indexed passages, for logging.
func (ix *TitleIndex) Passages() int {
	n := 0
	for _, ids := range ix.ByTitle {
		n += len(ids)
	}
	return n
}

// Neighbours returns the ids of the passages that sit within radius positions
// of id inside its own article, nearest first, excluding id itself.
//
// Nearest first matters: with a cap on the context, the passage immediately
// before or after the hit is worth more than one two positions away, so it
// must not be the one that gets dropped.
func (ix *TitleIndex) Neighbours(title string, id uint64, radius int) []uint64 {
	ids := ix.ByTitle[title]
	pos, ok := slices.BinarySearch(ids, id)
	if !ok {
		return nil
	}

	var out []uint64
	for d := 1; d <= radius; d++ {
		if pos-d >= 0 {
			out = append(out, ids[pos-d])
		}
		if pos+d < len(ids) {
			out = append(out, ids[pos+d])
		}
	}
	return out
}
