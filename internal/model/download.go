package model

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/schollz/progressbar/v3"

	"github.com/Aayush9029/peel/internal/config"
)

// DownloadIfMissing returns the path to the cached model, downloading it from
// the source URL if the file is absent. Set verify=true to MD5-check after
// download (always done during initial fetch; skipped on cache hit so we don't
// re-read 973 MB on every invocation).
func DownloadIfMissing(m config.Model, force bool, verify bool) (string, error) {
	path, err := config.ModelPath(m)
	if err != nil {
		return "", err
	}

	if !force {
		if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
			return path, nil
		}
	}

	if _, err := config.EnsureCacheDir(); err != nil {
		return "", err
	}

	if err := download(m.URL, path, fmt.Sprintf("downloading %s", m.Filename)); err != nil {
		return "", err
	}

	if verify && m.MD5 != "" {
		got, err := md5File(path)
		if err != nil {
			return "", fmt.Errorf("verify checksum: %w", err)
		}
		if got != m.MD5 {
			os.Remove(path)
			return "", fmt.Errorf("checksum mismatch: got %s want %s (re-run `peel download`)", got, m.MD5)
		}
	}

	return path, nil
}

// download streams `url` to `dst` via a tmp file + atomic rename, with a
// progress bar to stderr. The tmp-file shuffle prevents a corrupted partial
// from masquerading as a cached model on the next run.
func download(url, dst, desc string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("http get: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}

	bar := progressbar.NewOptions64(
		resp.ContentLength,
		progressbar.OptionSetDescription(desc),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowBytes(true),
		progressbar.OptionSetWidth(30),
		progressbar.OptionThrottle(100*1e6), // 100ms
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() { fmt.Fprintln(os.Stderr) }),
		progressbar.OptionSetRenderBlankState(true),
	)

	if _, err := io.Copy(io.MultiWriter(out, bar), resp.Body); err != nil {
		out.Close()
		os.Remove(tmp)
		return fmt.Errorf("write: %w", err)
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return err
	}
	// Best-effort: ensure intermediate dirs exist (Rename above would've
	// failed otherwise — but defensive against partial bugs in CacheDir).
	_ = os.MkdirAll(filepath.Dir(dst), 0o755)
	return nil
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
