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

Cloud background removers eat your originals. Vision API and U²-Net erode product photos. `peel` runs BiRefNet-General locally and produces clean alpha cutouts — the same model rembg ships, wrapped in a single Go binary with no Python in sight.

## Install

```bash
brew install aayush9029/tap/peel
```

## Usage

```bash
peel photo.jpg                # writes ./photo.png
peel photo.jpg -o out.png     # explicit name (overwrites)
peel *.webp                   # batch into cwd
peel *.webp -o out/           # batch into a directory
```

Reads JPG, PNG, WEBP. Writes PNG with alpha. Model auto-downloads on first run into `~/.config/peel/models/`. If `./photo.png` already exists, the next run writes `peel0123-photo.png` instead of clobbering it.

## License

MIT
