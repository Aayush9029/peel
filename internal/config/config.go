package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

type Model struct {
	ID       string
	Filename string
	URL      string
	SizeMB   int
	MD5      string
}

// Pinned to the rembg v0.0.0 artefact — that's the build proven on the
// medication-asset batch. Newer revisions reorganise the output node.
var BiRefNetGeneral = Model{
	ID:       "birefnet-general",
	Filename: "birefnet-general.onnx",
	URL:      "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
	SizeMB:   973,
	MD5:      "7a35a0141cbbc80de11d9c9a28f52697",
}

func CacheDir() (string, error) {
	if dir := os.Getenv("PEEL_CACHE_DIR"); dir != "" {
		return dir, nil
	}
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "peel", "models"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "peel", "models"), nil
}

func ModelPath(m Model) (string, error) {
	dir, err := CacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, m.Filename), nil
}

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

// OnnxRuntimeLibrary probes the standard Homebrew install paths; the formula's
// `depends_on "onnxruntime"` guarantees one will exist on user machines.
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
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
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
