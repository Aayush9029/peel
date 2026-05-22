package birefnet

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"golang.org/x/image/draw"
	ort "github.com/yalue/onnxruntime_go"

	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"
)

const (
	inputSize  = 1024
	inputName  = "input_image"
	outputName = "output_image"
)

// ImageNet normalisation — BiRefNet's Swin backbone was trained with these.
var (
	mean = [3]float32{0.485, 0.456, 0.406}
	std  = [3]float32{0.229, 0.224, 0.225}
)

var supportedExts = []string{".jpg", ".jpeg", ".png", ".webp"}

func IsImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	for _, e := range supportedExts {
		if e == ext {
			return true
		}
	}
	return false
}

func SupportedFormats() string { return strings.Join(supportedExts, ", ") }

var (
	envOnce sync.Once
	envErr  error
)

func initEnv() error {
	envOnce.Do(func() {
		var candidates []string
		switch runtime.GOOS {
		case "darwin":
			candidates = []string{
				"/opt/homebrew/lib/libonnxruntime.dylib",
				"/usr/local/lib/libonnxruntime.dylib",
			}
		case "linux":
			candidates = []string{
				"/usr/lib/libonnxruntime.so",
				"/usr/local/lib/libonnxruntime.so",
			}
		}
		var lib string
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				lib = c
				break
			}
		}
		if lib == "" {
			envErr = fmt.Errorf("ONNX Runtime not found; install with `brew install onnxruntime`")
			return
		}
		ort.SetSharedLibraryPath(lib)
		if err := ort.InitializeEnvironment(); err != nil {
			envErr = err
		}
	})
	return envErr
}

type Session struct {
	inner *ort.AdvancedSession
	in    *ort.Tensor[float32]
	out   *ort.Tensor[float32]
}

func NewSession(modelPath string) (*Session, error) {
	if err := initEnv(); err != nil {
		return nil, err
	}
	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, inputSize, inputSize))
	if err != nil {
		return nil, err
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, inputSize, inputSize))
	if err != nil {
		in.Destroy()
		return nil, err
	}
	inner, err := ort.NewAdvancedSession(modelPath,
		[]string{inputName}, []string{outputName},
		[]ort.Value{in}, []ort.Value{out}, nil)
	if err != nil {
		in.Destroy()
		out.Destroy()
		return nil, err
	}
	return &Session{inner: inner, in: in, out: out}, nil
}

func (s *Session) Close() {
	s.inner.Destroy()
	s.in.Destroy()
	s.out.Destroy()
}

func (s *Session) Process(inPath, outPath string) error {
	src, err := decode(inPath)
	if err != nil {
		return fmt.Errorf("decode %s: %w", inPath, err)
	}
	rgba, err := s.removeBackground(src)
	if err != nil {
		return err
	}
	return writePNG(outPath, rgba)
}

func (s *Session) removeBackground(src image.Image) (*image.NRGBA, error) {
	bounds := src.Bounds()
	srcRGBA := image.NewNRGBA(bounds)
	draw.Copy(srcRGBA, bounds.Min, src, bounds, draw.Src, nil)

	// CatmullRom stands in for Pillow's LANCZOS — visually equivalent on the
	// medication-asset run and the closest the x/image/draw API gets.
	resized := image.NewNRGBA(image.Rect(0, 0, inputSize, inputSize))
	draw.CatmullRom.Scale(resized, resized.Bounds(), srcRGBA, bounds, draw.Src, nil)

	// rembg parity: divide by max(pixel, 1e-6), NOT by 255.
	var pxMax uint8
	for i := 0; i < len(resized.Pix); i += 4 {
		if resized.Pix[i] > pxMax {
			pxMax = resized.Pix[i]
		}
		if resized.Pix[i+1] > pxMax {
			pxMax = resized.Pix[i+1]
		}
		if resized.Pix[i+2] > pxMax {
			pxMax = resized.Pix[i+2]
		}
	}
	maxF := float32(pxMax)
	if maxF < 1e-6 {
		maxF = 1e-6
	}

	data := s.in.GetData()
	const HW = inputSize * inputSize
	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			pix := (y*inputSize + x) * 4
			r := float32(resized.Pix[pix]) / maxF
			g := float32(resized.Pix[pix+1]) / maxF
			b := float32(resized.Pix[pix+2]) / maxF
			data[0*HW+y*inputSize+x] = (r - mean[0]) / std[0]
			data[1*HW+y*inputSize+x] = (g - mean[1]) / std[1]
			data[2*HW+y*inputSize+x] = (b - mean[2]) / std[2]
		}
	}

	if err := s.inner.Run(); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	raw := s.out.GetData()
	mn, mx := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	maskF := make([]float32, HW)
	for i := 0; i < HW; i++ {
		v := float32(1.0 / (1.0 + math.Exp(float64(-raw[i]))))
		maskF[i] = v
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if span := mx - mn; span > 1e-6 {
		for i := range maskF {
			maskF[i] = (maskF[i] - mn) / span
		}
	}

	maskImg := image.NewGray(image.Rect(0, 0, inputSize, inputSize))
	for i, v := range maskF {
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
	w, h := bounds.Dx(), bounds.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			so := y*srcRGBA.Stride + x*4
			do := y*out.Stride + x*4
			out.Pix[do] = srcRGBA.Pix[so]
			out.Pix[do+1] = srcRGBA.Pix[so+1]
			out.Pix[do+2] = srcRGBA.Pix[so+2]
			out.Pix[do+3] = maskFull.Pix[y*maskFull.Stride+x]
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
	if err == image.ErrFormat {
		return nil, fmt.Errorf("unsupported image format (peel reads %s)", SupportedFormats())
	}
	return img, err
}

func writePNG(path string, img image.Image) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
