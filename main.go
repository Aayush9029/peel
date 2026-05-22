package main

import (
	"bufio"
	"errors"
	"fmt"
	"math/rand"
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
	case "delete", "rm":
		return cmdDelete(args[1:])
	}
	return cmdProcess(args)
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

	// Validate inputs up front so typos surface before the model load.
	for _, in := range inputs {
		if _, err := os.Stat(in); err != nil {
			return fmt.Errorf("input %s: %w", in, err)
		}
		if !bgremove.IsLikelyImage(in) {
			return fmt.Errorf("input %s: unsupported extension (peel reads %s)", in, strings.Join(bgremove.SupportedInputExts, ", "))
		}
	}

	outputs, err := resolveOutputs(inputs, opts.output)
	if err != nil {
		return err
	}

	m := config.BiRefNetGeneral
	modelPath, err := config.ModelPath(m)
	if err != nil {
		return err
	}
	if fi, statErr := os.Stat(modelPath); statErr != nil || fi.Size() == 0 {
		if opts.noDownload {
			return fmt.Errorf("model not found at %s and --no-download was set", modelPath)
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

// resolveOutputs maps inputs to PNG paths. `-o` is authoritative: file when
// single input, directory when batch or trailing-slash. Without `-o`, default
// to <base>.png in cwd, swapping to peel<4-digit>-<base>.png on collision so
// we never silently overwrite an existing file.
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
		if dir := filepath.Dir(outFlag); dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, err
			}
		}
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

type processOpts struct {
	output     string
	noDownload bool
}

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

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
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
  peel a.jpg                    Write ./a.png (auto peel0123-a.png on collision)
  peel a.jpg -o out.png         Explicit output (overwrites)
  peel a.jpg b.jpg c.webp       Batch into cwd
  peel *.webp -o out/           Batch into a directory
  peel delete [-y]              Remove the cached model

%sFlags:%s
  -o, --output <path>           Output file (single) or directory (batch)
      --no-download             Fail instead of auto-downloading the model
  -h, --help                    Show this help
  -v, --version                 Show the version

%sFormats:%s JPG, PNG, WEBP in → PNG with alpha out.
%sModel:%s    ~/.config/peel/models/  (override with PEEL_CACHE_DIR or XDG_CONFIG_HOME)
`,
		ui.Bold, ui.Reset, version,
		ui.Bold, ui.Reset,
		ui.Bold, ui.Reset,
		ui.Bold, ui.Reset,
		ui.Bold, ui.Reset,
	)
}
