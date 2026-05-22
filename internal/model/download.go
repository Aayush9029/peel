package model

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/schollz/progressbar/v3"

	"github.com/Aayush9029/peel/internal/config"
)

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
			return "", fmt.Errorf("checksum mismatch: got %s want %s", got, m.MD5)
		}
	}

	return path, nil
}

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
		progressbar.OptionThrottle(100*1e6),
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
	return os.Rename(tmp, dst)
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
