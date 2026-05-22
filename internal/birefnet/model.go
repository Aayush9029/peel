package birefnet

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/schollz/progressbar/v3"
)

const (
	modelURL      = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"
	modelMD5      = "7a35a0141cbbc80de11d9c9a28f52697"
	modelFilename = "birefnet-general.onnx"
	ModelSizeMB   = 973
)

func ModelPath() (string, error) {
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "peel", "models", modelFilename), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "peel", "models", modelFilename), nil
}

func DownloadModel() (string, error) {
	dst, err := ModelPath()
	if err != nil {
		return "", err
	}
	if fi, err := os.Stat(dst); err == nil && fi.Size() > 0 {
		return dst, nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", err
	}

	resp, err := http.Get(modelURL)
	if err != nil {
		return "", fmt.Errorf("http get: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return "", fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return "", err
	}
	bar := progressbar.NewOptions64(
		resp.ContentLength,
		progressbar.OptionSetDescription("downloading "+modelFilename),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowBytes(true),
		progressbar.OptionSetWidth(30),
		progressbar.OptionThrottle(100*1e6),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() { fmt.Fprintln(os.Stderr) }),
	)
	if _, err := io.Copy(io.MultiWriter(out, bar), resp.Body); err != nil {
		out.Close()
		os.Remove(tmp)
		return "", err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return "", err
	}
	if err := os.Rename(tmp, dst); err != nil {
		return "", err
	}

	got, err := md5File(dst)
	if err != nil {
		return "", err
	}
	if got != modelMD5 {
		os.Remove(dst)
		return "", fmt.Errorf("checksum mismatch: got %s want %s", got, modelMD5)
	}
	return dst, nil
}

func md5File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := md5.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
