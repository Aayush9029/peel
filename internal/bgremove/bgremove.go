// Package bgremove turns input image bytes into transparent-background RGBA
// PNGs by routing them through a BiRefNet session.
package bgremove

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/draw"

	// Register decoders for the formats peel accepts as input. JPEG and
	// PNG ship in stdlib; WebP needs the x/image extension.
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/Aayush9029/peel/internal/session"
)

// SupportedInputExts lists the file extensions peel will attempt to decode.
// Used for arg-validation messages — the actual decode path is format-sniffed
// via image.Decode so unusual extensions still work if the bytes are valid.
var SupportedInputExts = []string{".jpg", ".jpeg", ".png", ".webp"}

// IsLikelyImage returns true for paths whose extension peel recognises. The
// stdlib decoder will still reject a bad payload — this is just for fast
// pre-flight validation.
func IsLikelyImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	for _, e := range SupportedInputExts {
		if e == ext {
			return true
		}
	}
	return false
}

// ProcessFile reads `inPath`, removes its background, and writes a transparent
// PNG to `outPath`. The output extension is forced to .png regardless of what
// the caller supplied — alpha cutouts in JPEG are an oxymoron.
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

// RemoveBackground returns a new RGBA image where pixels predicted as
// background carry alpha 0 and foreground pixels keep their original color
// premultiplied by the mask.
func RemoveBackground(sess *session.Session, src image.Image) (*image.NRGBA, error) {
	bounds := src.Bounds()
	srcRGBA := image.NewNRGBA(bounds)
	draw.Copy(srcRGBA, bounds.Min, src, bounds, draw.Src, nil)

	// Resize to 1024×1024 for inference. CatmullRom is a reasonable
	// LANCZOS stand-in for the x/image/draw API and matches the quality
	// rembg's Pillow LANCZOS produced in our medication-asset run.
	resized := image.NewNRGBA(image.Rect(0, 0, session.InputSize, session.InputSize))
	draw.CatmullRom.Scale(resized, resized.Bounds(), srcRGBA, bounds, draw.Src, nil)

	// Pack to row-major RGB uint8s — the session converts to float32 +
	// CHW internally.
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

	// Convert flat mask → 8-bit grayscale image so we can resize it with
	// the same draw machinery.
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

	// Compose: copy source RGB, alpha = mask.
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
		// Provide a clearer message than image.Decode's terse default.
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

// AlphaStats summarises the alpha channel of an RGBA image. Used by E2E tests
// to assert that background removal actually produced transparency.
type AlphaStats struct {
	Total        int
	Transparent  int // alpha == 0
	Opaque       int // alpha == 255
	MeanAlpha    float64
}

// ComputeAlphaStats walks every pixel and tallies alpha statistics. Cheap
// enough to run on test-sized outputs.
func ComputeAlphaStats(img image.Image) AlphaStats {
	bounds := img.Bounds()
	var s AlphaStats
	var sum int64
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			// RGBA() returns 16-bit values; scale back to 8-bit for clarity.
			a8 := uint8(a >> 8)
			if a8 == 0 {
				s.Transparent++
			} else if a8 == 255 {
				s.Opaque++
			}
			sum += int64(a8)
			s.Total++
		}
	}
	if s.Total > 0 {
		s.MeanAlpha = float64(sum) / float64(s.Total)
	}
	return s
}

// Ensure that we don't accidentally import unused stdlib pieces — silences
// staticcheck noise during early development.
var _ io.Reader = (*os.File)(nil)
var _ color.Color = color.NRGBA{}
