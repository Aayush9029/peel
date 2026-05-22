// peel removes backgrounds from images using the BiRefNet-General ONNX model.
//
// Usage:
//
//	peel download              # fetch the model into the cache
//	peel list                  # show installed models + sizes + paths
//	peel delete [-y]           # remove cached model(s)
//	peel a.jpg -o b.png        # single image
//	peel a.jpg b.jpg -o out/   # batch into a directory
package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Aayush9029/peel/internal/bgremove"
	"github.com/Aayush9029/peel/internal/config"
	"github.com/Aayush9029/peel/internal/model"
	"github.com/Aayush9029/peel/internal/session"
	"github.com/Aayush9029/peel/internal/ui"
)

var version = "dev"

func main() {
	if err := run(os.Args[1:]); err != nil {
		ui.Fatalf("%s", err)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printHelp()
		return nil
	}
	switch args[0] {
	case "-h", "--help", "help":
		printHelp()
		return nil
	case "-v", "--version", "version":
		fmt.Printf("peel %s\n", version)
		return nil
	case "download":
		return cmdDownload(args[1:])
	case "list":
		return cmdList(args[1:])
	case "delete", "rm":
		return cmdDelete(args[1:])
	}
	// Anything else → treat as image processing.
	return cmdProcess(args)
}

func cmdDownload(args []string) error {
	force := false
	for _, a := range args {
		switch a {
		case "-f", "--force":
			force = true
		case "-h", "--help":
			fmt.Println("usage: peel download [--force]")
			return nil
		default:
			return fmt.Errorf("unknown flag for download: %s", a)
		}
	}
	m := config.BiRefNetGeneral
	path, err := config.ModelPath(m)
	if err != nil {
		return err
	}
	if !force {
		if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
			ui.Success(fmt.Sprintf("already cached: %s", path))
			return nil
		}
	}
	ui.Status(fmt.Sprintf("downloading %s (~%d MB) → %s", m.ID, m.SizeMB, path))
	out, err := model.DownloadIfMissing(m, force, true)
	if err != nil {
		return err
	}
	ui.Success(fmt.Sprintf("saved: %s", out))
	return nil
}

func cmdList(args []string) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		fmt.Println("usage: peel list")
		return nil
	}
	dir, err := config.CacheDir()
	if err != nil {
		return err
	}
	ui.Dimf("cache: %s", dir)
	fmt.Println()

	models := []config.Model{config.BiRefNetGeneral}
	for _, m := range models {
		path, _ := config.ModelPath(m)
		fi, err := os.Stat(path)
		status := "not installed"
		size := ""
		if err == nil {
			status = "installed"
			size = humanizeBytes(fi.Size())
		}
		fmt.Printf("  %-22s  %-12s  %s\n", m.ID, status, size)
		ui.Dimf("    %s", path)
	}
	return nil
}

func cmdDelete(args []string) error {
	yes := false
	for _, a := range args {
		switch a {
		case "-y", "--yes":
			yes = true
		case "-h", "--help":
			fmt.Println("usage: peel delete [-y]")
			return nil
		default:
			return fmt.Errorf("unknown flag for delete: %s", a)
		}
	}
	m := config.BiRefNetGeneral
	path, err := config.ModelPath(m)
	if err != nil {
		return err
	}
	fi, err := os.Stat(path)
	if err != nil {
		ui.Status(fmt.Sprintf("nothing to delete (%s does not exist)", path))
		return nil
	}
	if !yes {
		fmt.Printf("Delete %s (%s)? [y/N] ", path, humanizeBytes(fi.Size()))
		reader := bufio.NewReader(os.Stdin)
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(strings.ToLower(line))
		if line != "y" && line != "yes" {
			ui.Status("aborted")
			return nil
		}
	}
	if err := os.Remove(path); err != nil {
		return err
	}
	ui.Success(fmt.Sprintf("removed %s", path))
	return nil
}

func cmdProcess(args []string) error {
	opts, inputs, err := parseProcessFlags(args)
	if err != nil {
		return err
	}
	if len(inputs) == 0 {
		return errors.New("no input images (try `peel --help`)")
	}

	// Validate inputs up front so users find typos before we eat the
	// ~6-second model load on Apple Silicon.
	for _, in := range inputs {
		if _, err := os.Stat(in); err != nil {
			return fmt.Errorf("input %s: %w", in, err)
		}
		if !bgremove.IsLikelyImage(in) {
			return fmt.Errorf("input %s: unsupported extension (peel reads %s)", in, strings.Join(bgremove.SupportedInputExts, ", "))
		}
	}

	// Decide output paths. Single-input: `-o` is a file (or auto if absent).
	// Multi-input: `-o` is a directory (or default to ./).
	outputs := make([]string, len(inputs))
	if len(inputs) == 1 {
		if opts.output == "" {
			outputs[0] = swapExt(inputs[0], ".png")
		} else if isDir(opts.output) || strings.HasSuffix(opts.output, "/") {
			if err := os.MkdirAll(opts.output, 0o755); err != nil {
				return err
			}
			outputs[0] = filepath.Join(opts.output, filepath.Base(swapExt(inputs[0], ".png")))
		} else {
			outputs[0] = opts.output
		}
	} else {
		dir := opts.output
		if dir == "" {
			dir = "."
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		for i, in := range inputs {
			outputs[i] = filepath.Join(dir, filepath.Base(swapExt(in, ".png")))
		}
	}

	// Ensure the model is on disk. Auto-download unless --no-download.
	m := config.BiRefNetGeneral
	modelPath, err := config.ModelPath(m)
	if err != nil {
		return err
	}
	if fi, statErr := os.Stat(modelPath); statErr != nil || fi.Size() == 0 {
		if opts.noDownload {
			return fmt.Errorf("model not found at %s and --no-download was set (run `peel download`)", modelPath)
		}
		ui.Status(fmt.Sprintf("downloading model (~%d MB) → %s", m.SizeMB, modelPath))
		if _, err := model.DownloadIfMissing(m, false, true); err != nil {
			return err
		}
	}

	ui.Status(fmt.Sprintf("loading model (%s)", filepath.Base(modelPath)))
	loadStart := time.Now()
	sess, err := session.New(modelPath)
	if err != nil {
		return err
	}
	defer sess.Close()
	ui.Dimf("  loaded in %s", time.Since(loadStart).Round(100*time.Millisecond))

	for i, in := range inputs {
		start := time.Now()
		ui.Status(fmt.Sprintf("%s → %s", in, outputs[i]))
		if err := bgremove.ProcessFile(sess, in, outputs[i]); err != nil {
			return err
		}
		ui.Success(fmt.Sprintf("  done in %s", time.Since(start).Round(100*time.Millisecond)))
	}
	return nil
}

type processOpts struct {
	output     string
	noDownload bool
}

// parseProcessFlags pulls -o/--output and --no-download out of args, leaving
// the rest as positional input paths. Flags can appear before, after, or
// interleaved with positionals — `peel a.jpg -o b.png c.jpg` works.
func parseProcessFlags(args []string) (processOpts, []string, error) {
	var opts processOpts
	var positional []string

	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-o" || a == "--output":
			if i+1 >= len(args) {
				return opts, nil, fmt.Errorf("%s requires a value", a)
			}
			opts.output = args[i+1]
			i++
		case strings.HasPrefix(a, "--output="):
			opts.output = strings.TrimPrefix(a, "--output=")
		case strings.HasPrefix(a, "-o="):
			opts.output = strings.TrimPrefix(a, "-o=")
		case a == "--no-download":
			opts.noDownload = true
		case a == "-h" || a == "--help":
			printHelp()
			os.Exit(0)
		case strings.HasPrefix(a, "-"):
			return opts, nil, fmt.Errorf("unknown flag: %s", a)
		default:
			positional = append(positional, a)
		}
	}
	return opts, positional, nil
}

func swapExt(path, newExt string) string {
	ext := filepath.Ext(path)
	return strings.TrimSuffix(path, ext) + newExt
}

func isDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}

func humanizeBytes(n int64) string {
	const u = 1024
	if n < u {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(u), 0
	for v := n / u; v >= u; v /= u {
		div *= u
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func printHelp() {
	fmt.Printf(`%speel%s v%s — remove image backgrounds with BiRefNet

%sUsage:%s
  peel download                 Fetch the BiRefNet model into the cache.
  peel list                     Show installed models, sizes, and paths.
  peel delete [-y]              Remove cached model(s).
  peel <image> [-o <out.png>]   Remove background from a single image.
  peel a.jpg b.jpg -o out/      Batch: -o is the output directory.

%sFlags:%s
  -o, --output <path>           Output file (single) or directory (batch).
      --no-download             Fail instead of auto-downloading the model.
  -h, --help                    Show this help.
  -v, --version                 Show the version.

%sFormats:%s JPG, PNG, WEBP in → PNG with alpha out.
%sCache:%s    ~/.cache/peel/  (override with PEEL_CACHE_DIR or XDG_CACHE_HOME)
`,
		ui.Bold, ui.Reset, version,
		ui.Bold, ui.Reset,
		ui.Bold, ui.Reset,
		ui.Bold, ui.Reset,
		ui.Bold, ui.Reset,
	)
}
