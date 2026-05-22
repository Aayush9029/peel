package bgremove

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/draw"

	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/Aayush9029/peel/internal/session"
)

var SupportedInputExts = []string{".jpg", ".jpeg", ".png", ".webp"}

func IsLikelyImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	for _, e := range SupportedInputExts {
		if e == ext {
			return true
		}
	}
	return false
}

func ProcessFile(sess *session.Session, inPath, outPath string) error {
	src, err := decode(inPath)
	if err != nil {
		return fmt.Errorf("decode %s: %w", inPath, err)
	}
	out, err := RemoveBackground(sess, src)
	if err != nil {
		return err
	}
	return writePNG(outPath, out)
}

func RemoveBackground(sess *session.Session, src image.Image) (*image.NRGBA, error) {
	bounds := src.Bounds()
	srcRGBA := image.NewNRGBA(bounds)
	draw.Copy(srcRGBA, bounds.Min, src, bounds, draw.Src, nil)

	// CatmullRom stands in for Pillow's LANCZOS — visually equivalent on the
	// medication-asset run and the closest the x/image/draw API gets.
	resized := image.NewNRGBA(image.Rect(0, 0, session.InputSize, session.InputSize))
	draw.CatmullRom.Scale(resized, resized.Bounds(), srcRGBA, bounds, draw.Src, nil)

	rgb := make([]uint8, session.InputSize*session.InputSize*3)
	for y := 0; y < session.InputSize; y++ {
		for x := 0; x < session.InputSize; x++ {
			idx := (y*session.InputSize + x) * 4
			out := (y*session.InputSize + x) * 3
			rgb[out] = resized.Pix[idx]
			rgb[out+1] = resized.Pix[idx+1]
			rgb[out+2] = resized.Pix[idx+2]
		}
	}

	mask, err := sess.Predict(rgb)
	if err != nil {
		return nil, err
	}

	maskImg := image.NewGray(image.Rect(0, 0, session.InputSize, session.InputSize))
	for i, v := range mask {
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		maskImg.Pix[i] = uint8(v*255 + 0.5)
	}
	maskFull := image.NewGray(bounds)
	draw.CatmullRom.Scale(maskFull, bounds, maskImg, maskImg.Bounds(), draw.Src, nil)

	out := image.NewNRGBA(bounds)
	w := bounds.Dx()
	h := bounds.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcOff := y*srcRGBA.Stride + x*4
			dstOff := y*out.Stride + x*4
			out.Pix[dstOff] = srcRGBA.Pix[srcOff]
			out.Pix[dstOff+1] = srcRGBA.Pix[srcOff+1]
			out.Pix[dstOff+2] = srcRGBA.Pix[srcOff+2]
			out.Pix[dstOff+3] = maskFull.Pix[y*maskFull.Stride+x]
		}
	}
	return out, nil
}

func decode(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		if err == image.ErrFormat {
			return nil, fmt.Errorf("unsupported image format (peel reads JPG, PNG, WEBP)")
		}
		return nil, err
	}
	return img, nil
}

func writePNG(path string, img image.Image) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := &png.Encoder{CompressionLevel: png.DefaultCompression}
	return enc.Encode(f, img)
}
