# peel

`peel` is a Go CLI that removes image backgrounds on the Mac with the BiRefNet-General ONNX model. It reads JPG, PNG, and WEBP, and writes PNG with alpha. It ships through the Homebrew tap `aayush9029/tap`. Status: shipped.

## Commands

| Task | Command |
|------|---------|
| Build | `go build -ldflags "-s -w -X main.version=dev" -o peel .` |
| Run | `go run . photo.jpg -o out.png` |
| Release | Load the `oss-cli-release` skill |

## Architecture

- `main.go`: arguments, output names, and batch mode.
- `internal/birefnet`: model download (`model.go`) and inference through `github.com/yalue/onnxruntime_go` (`birefnet.go`).
- The model downloads on the first run from the rembg GitHub release into `~/.config/peel/models/` (or `$XDG_CONFIG_HOME/peel/models/`).

## Rules

- `peel` needs the ONNX Runtime shared library. It looks in `/opt/homebrew/lib` and `/usr/local/lib`. The tap formula depends on `onnxruntime`. For local work, run `brew install onnxruntime`.
- Never overwrite an input or an existing output when no `-o` is given. If `./photo.png` exists, write `peel<NNNN>-photo.png`.
- Keep the CLI small. Recent work removed the `list` and `download` subcommands.
- The tap formula `../../homebrew-tap/Formula/peel.rb` builds from the tag tarball with `-X main.version=#{version}`.
