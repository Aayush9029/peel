<p align="center">
  <img src="assets/icon.png" width="128" alt="peel">
  <h1 align="center">peel</h1>
  <p align="center">Remove image backgrounds in the terminal with BiRefNet</p>
</p>

<p align="center">
  <a href="https://github.com/Aayush9029/peel/releases/latest"><img src="https://img.shields.io/github/v/release/Aayush9029/peel" alt="Release"></a>
  <a href="https://github.com/Aayush9029/peel/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Aayush9029/peel" alt="License"></a>
</p>

## Why

Cloud background removers eat your originals. Vision API and U²-Net erode product photos. `peel` runs BiRefNet-General locally and produces clean alpha cutouts — the same model rembg ships, wrapped in a single static Go binary with no Python in sight.

## Install

```bash
brew install aayush9029/tap/peel
```

Or tap first:

```bash
brew tap aayush9029/tap
brew install peel
```

## Usage

```bash
peel download                       # fetch the model (~973 MB, one time)
peel list                           # show installed models
peel photo.jpg -o photo.png         # remove background from one image
peel a.jpg b.jpg c.webp -o out/     # batch into a directory
peel delete -y                      # free up disk
```

Input: JPG, PNG, WEBP. Output: PNG with alpha. The model auto-downloads on first inference if missing — pass `--no-download` to fail instead.

## License

MIT
