package main

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Aayush9029/peel/internal/birefnet"
)

var version = "dev"

const (
	cReset = "\x1b[0m"
	cBold  = "\x1b[1m"
	cDim   = "\x1b[2m"
	cRed   = "\x1b[31m"
	cGreen = "\x1b[32m"
	cCyan  = "\x1b[36m"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "%s✗%s %s\n", cRed, cReset, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printHelp()
		return nil
	}
	switch args[0] {
	case "-h", "--help":
		printHelp()
		return nil
	case "-v", "--version":
		fmt.Printf("peel %s\n", version)
		return nil
	}

	output, inputs, err := parseArgs(args)
	if err != nil {
		return err
	}
	if len(inputs) == 0 {
		return errors.New("no input images (try `peel --help`)")
	}

	for _, in := range inputs {
		if _, err := os.Stat(in); err != nil {
			return fmt.Errorf("input %s: %w", in, err)
		}
		if !birefnet.IsImage(in) {
			return fmt.Errorf("input %s: unsupported (peel reads %s)", in, birefnet.SupportedFormats())
		}
	}

	outputs, err := resolveOutputs(inputs, output)
	if err != nil {
		return err
	}

	modelPath, err := birefnet.ModelPath()
	if err != nil {
		return err
	}
	if fi, err := os.Stat(modelPath); err != nil || fi.Size() == 0 {
		fmt.Fprintf(os.Stderr, "%s→%s downloading model (~%d MB) → %s\n", cCyan, cReset, birefnet.ModelSizeMB, modelPath)
		if _, err := birefnet.DownloadModel(); err != nil {
			return err
		}
	}

	loadStart := time.Now()
	sess, err := birefnet.NewSession(modelPath)
	if err != nil {
		return err
	}
	defer sess.Close()
	fmt.Printf("%sloaded in %s%s\n", cDim, time.Since(loadStart).Round(100*time.Millisecond), cReset)

	for i, in := range inputs {
		start := time.Now()
		if err := sess.Process(in, outputs[i]); err != nil {
			return err
		}
		fmt.Printf("%s✓%s %s → %s %s(%s)%s\n",
			cGreen, cReset, in, outputs[i],
			cDim, time.Since(start).Round(100*time.Millisecond), cReset)
	}
	return nil
}

// resolveOutputs maps inputs to PNG paths. `-o` is authoritative. Without it,
// default to <base>.png in cwd; on collision fall back to peel<4-digit>-<base>.png
// so re-runs never clobber prior work.
func resolveOutputs(inputs []string, outFlag string) ([]string, error) {
	outputs := make([]string, len(inputs))

	if outFlag != "" {
		treatAsDir := len(inputs) > 1 || isDir(outFlag) || strings.HasSuffix(outFlag, string(os.PathSeparator))
		if treatAsDir {
			if err := os.MkdirAll(outFlag, 0o755); err != nil {
				return nil, err
			}
			for i, in := range inputs {
				outputs[i] = filepath.Join(outFlag, filepath.Base(swapExt(in, ".png")))
			}
			return outputs, nil
		}
		outputs[0] = outFlag
		return outputs, nil
	}

	claimed := map[string]bool{}
	for i, in := range inputs {
		base := filepath.Base(swapExt(in, ".png"))
		candidate := base
		if claimed[candidate] || fileExists(candidate) {
			candidate = fmt.Sprintf("peel%04d-%s", rand.Intn(10000), base)
		}
		claimed[candidate] = true
		outputs[i] = candidate
	}
	return outputs, nil
}

func parseArgs(args []string) (output string, inputs []string, err error) {
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-o" || a == "--output":
			if i+1 >= len(args) {
				return "", nil, fmt.Errorf("%s requires a value", a)
			}
			output = args[i+1]
			i++
		case a == "-h" || a == "--help":
			printHelp()
			os.Exit(0)
		case strings.HasPrefix(a, "-"):
			return "", nil, fmt.Errorf("unknown flag: %s", a)
		default:
			inputs = append(inputs, a)
		}
	}
	return output, inputs, nil
}

func swapExt(path, newExt string) string {
	return strings.TrimSuffix(path, filepath.Ext(path)) + newExt
}

func isDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func printHelp() {
	fmt.Printf(`%speel%s v%s — remove image backgrounds

%sUsage:%s
  peel a.jpg                Write ./a.png (peel0123-a.png on collision)
  peel a.jpg -o out.png     Explicit name (overwrites)
  peel *.webp               Batch into cwd
  peel *.webp -o out/       Batch into a directory

Reads JPG, PNG, WEBP. Writes PNG with alpha.
Model lives in ~/.config/peel/models/ (~973 MB, auto-downloads).
`, cBold, cReset, version, cBold, cReset)
}
