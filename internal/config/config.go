package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// Model is the static description of a downloadable ONNX model. Today peel
// ships exactly one (BiRefNet-general), but the type leaves room for adding
// alternatives without rewriting the cache/list/delete plumbing.
type Model struct {
	ID       string // stable identifier used in URLs + filenames
	Filename string // on-disk name inside the cache directory
	URL      string // direct .onnx download URL
	SizeMB   int    // expected download size, used in `peel list`
	MD5      string // optional integrity check
}

// BiRefNetGeneral is the one model peel knows how to drive. Pinned to the
// rembg v0.0.0 release because that's the artefact we proved on a 74-image
// medication batch — newer revisions reorganise the output node and break
// inference with no warning.
var BiRefNetGeneral = Model{
	ID:       "birefnet-general",
	Filename: "birefnet-general.onnx",
	URL:      "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
	SizeMB:   973,
	MD5:      "7a35a0141cbbc80de11d9c9a28f52697",
}

// CacheDir returns the directory peel uses to store ONNX models. Honors
// PEEL_CACHE_DIR for tests/overrides, otherwise defaults to XDG.
func CacheDir() (string, error) {
	if dir := os.Getenv("PEEL_CACHE_DIR"); dir != "" {
		return dir, nil
	}
	// Use XDG_DATA_HOME if set, else ~/.cache/peel — matches the rest of
	// the brew-tools family and keeps us out of rembg's ~/.u2net so each
	// CLI owns its cache.
	if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
		return filepath.Join(xdg, "peel"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".cache", "peel"), nil
}

// ModelPath returns where the given model would live on disk.
func ModelPath(m Model) (string, error) {
	dir, err := CacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, m.Filename), nil
}

// EnsureCacheDir creates the cache directory if it doesn't exist.
func EnsureCacheDir() (string, error) {
	dir, err := CacheDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

// OnnxRuntimeLibrary returns the path to libonnxruntime.dylib. Honors
// ONNXRUNTIME_LIB for tests/dev, otherwise probes the standard Homebrew
// install paths (the brew formula's `depends_on "onnxruntime"` guarantees
// it lives at one of these locations on user machines).
func OnnxRuntimeLibrary() (string, error) {
	if p := os.Getenv("ONNXRUNTIME_LIB"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
		return "", fmt.Errorf("ONNXRUNTIME_LIB=%s does not exist", p)
	}

	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",   // Apple Silicon
			"/usr/local/lib/libonnxruntime.dylib",      // Intel Mac
			"/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
			"/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib",
		}
	case "linux":
		candidates = []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
		}
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("could not locate the ONNX Runtime shared library; install it with `brew install onnxruntime` or set ONNXRUNTIME_LIB")
}
