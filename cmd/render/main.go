// Command render turns the JSON aggregates in eval/ into the LaTeX
// fragments the paper reads, next to it in paper/.
//
// It is the only place LaTeX is produced, and it is deliberately dumb: one
// eval/<name>.json becomes one paper/<name>.gen.tex, each
// name/value pair one \newcommand. All measurement lives upstream (the
// commands that write eval/), all presentation lives in the paper, and this
// step can be re-run in milliseconds - which is what makes the fragments a
// cache rather than data.
package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/mikolajsemeniuk/ragbench/pkg/eval"
)

func main() {
	var (
		evalDir = flag.String("eval", "eval", "directory holding the *.json aggregate documents")
		outDir  = flag.String("out", "paper", "directory to write the *.gen.tex fragments to")
	)
	flag.Parse()

	entries, err := os.ReadDir(*evalDir)
	if err != nil {
		log.Fatalf("reading %s: %v", *evalDir, err)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("creating %s: %v", *outDir, err)
	}

	rendered := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		source := filepath.Join(*evalDir, e.Name())
		doc, err := eval.Load(source)
		if err != nil {
			log.Fatalf("loading %s: %v", source, err)
		}
		base := strings.TrimSuffix(e.Name(), ".json")
		target := filepath.Join(*outDir, base+".gen.tex")
		if err := os.WriteFile(target, eval.RenderTex(doc, source), 0o644); err != nil {
			log.Fatalf("writing %s: %v", target, err)
		}
		rendered++
	}
	log.Printf("rendered %d fragments from %s into %s", rendered, *evalDir, *outDir)
}
