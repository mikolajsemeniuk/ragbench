// Package eval holds the JSON aggregate format that sits between the raw
// per-question dumps (runs/) and the LaTeX fragments the paper reads
// (paper/*.gen.tex).
//
// Every measuring command writes its aggregates here as ordered name/value
// pairs: the name is the LaTeX macro name, the value the exact text the
// macro expands to. cmd/render turns a document into a fragment. The split
// exists so that presentation is never welded to measurement - a fragment
// can be regenerated in milliseconds from eval/ without touching the GPU,
// while the JSON stays readable by anything that is not LaTeX.
//
// The pairs are a list, not a map, so that a rendered fragment is
// deterministic and can be diffed against the previous version.
package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Command is one aggregate: a LaTeX macro name (letters only) and the text
// it expands to. Values may span several lines - the summary tables are
// whole tabular environments stored under a single name.
type Command struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Doc is one eval file, rendered to one fragment.
type Doc struct {
	// Generator names the command whose measurement this is, e.g.
	// "cmd/bench". It is recorded in the rendered fragment's header so a
	// reader of the .tex still sees where a number came from.
	Generator string `json:"generator"`

	// Comments are documentation lines carried into the rendered fragment
	// as LaTeX comments, e.g. how to read a table's cells.
	Comments []string `json:"comments,omitempty"`

	Commands []Command `json:"commands"`
}

// Add appends one aggregate.
func (d *Doc) Add(name, value string) {
	d.Commands = append(d.Commands, Command{Name: name, Value: value})
}

// Addf appends one aggregate with fmt formatting.
func (d *Doc) Addf(name, format string, args ...any) {
	d.Add(name, fmt.Sprintf(format, args...))
}

// Write stores the document as indented JSON, creating directories as
// needed.
func Write(path string, d *Doc) error {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create dir: %w", err)
		}
	}

	raw, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal eval doc: %w", err)
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// Load reads a document written by Write.
func Load(path string) (*Doc, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d Doc
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &d, nil
}

// RenderTex renders the document as a LaTeX fragment. source names the eval
// file the fragment was rendered from, so the header points a reader back
// at the data.
func RenderTex(d *Doc, source string) []byte {
	var b bytes.Buffer
	fmt.Fprintf(&b, "%% generated automatically by cmd/render from %s (aggregated by %s) - do not edit by hand\n", source, d.Generator)
	for _, c := range d.Comments {
		fmt.Fprintf(&b, "%% %s\n", strings.TrimPrefix(c, "% "))
	}
	for _, c := range d.Commands {
		fmt.Fprintf(&b, "\\newcommand{\\%s}{%s}\n", c.Name, c.Value)
	}
	return b.Bytes()
}
