// Package session wraps an ONNX Runtime InferenceSession over BiRefNet.
// One *Session can be reused across many images — input/output tensors are
// allocated once and re-filled per Run().
package session

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/Aayush9029/peel/internal/config"
)

const (
	// InputSize is BiRefNet's expected square side (1024×1024 RGB).
	InputSize = 1024
)

// BiRefNet I/O node names — discovered via onnxruntime introspection of the
// distributed model. Hard-coded because re-querying the session metadata for
// every image is wasteful and the model is pinned.
const (
	inputName  = "input_image"
	outputName = "output_image"
)

// ImageNet normalisation. BiRefNet was trained against these because it sits
// on top of a Swin backbone — using the wrong values produces washed-out
// masks that look "almost right" until you check the alpha histogram.
var (
	mean = [3]float32{0.485, 0.456, 0.406}
	std  = [3]float32{0.229, 0.224, 0.225}
)

var (
	envOnce sync.Once
	envErr  error
)

// initEnv loads the ONNX Runtime shared library and bootstraps the global
// environment exactly once per process. Subsequent NewSession calls reuse it.
func initEnv() error {
	envOnce.Do(func() {
		libPath, err := config.OnnxRuntimeLibrary()
		if err != nil {
			envErr = err
			return
		}
		ort.SetSharedLibraryPath(libPath)
		if err := ort.InitializeEnvironment(); err != nil {
			envErr = fmt.Errorf("init onnxruntime: %w", err)
			return
		}
	})
	return envErr
}

// Session is a reusable BiRefNet inference handle. Not safe for concurrent
// use — wrap in a mutex or build one Session per goroutine.
type Session struct {
	inner       *ort.AdvancedSession
	inputTensor *ort.Tensor[float32]
	outputTens  *ort.Tensor[float32]
}

// New opens an ONNX session against the given .onnx file.
func New(modelPath string) (*Session, error) {
	if err := initEnv(); err != nil {
		return nil, err
	}

	inputShape := ort.NewShape(1, 3, InputSize, InputSize)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("alloc input tensor: %w", err)
	}

	outputShape := ort.NewShape(1, 1, InputSize, InputSize)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("alloc output tensor: %w", err)
	}

	inner, err := ort.NewAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("open onnx session: %w", err)
	}

	return &Session{
		inner:       inner,
		inputTensor: inputTensor,
		outputTens:  outputTensor,
	}, nil
}

// Close releases all native resources. Safe to call multiple times.
func (s *Session) Close() {
	if s == nil {
		return
	}
	if s.inner != nil {
		s.inner.Destroy()
		s.inner = nil
	}
	if s.inputTensor != nil {
		s.inputTensor.Destroy()
		s.inputTensor = nil
	}
	if s.outputTens != nil {
		s.outputTens.Destroy()
		s.outputTens = nil
	}
}

// Predict runs BiRefNet against a pre-resized 1024×1024 RGB image and returns
// the 1024×1024 grayscale alpha mask in row-major order (values 0..1).
// `rgbHWC` must be exactly 1024*1024*3 uint8s (R,G,B per pixel).
//
// Pre/post-processing matches rembg's BiRefNetSessionGeneral exactly so that
// users who calibrated their workflow on rembg get identical cutouts.
func (s *Session) Predict(rgbHWC []uint8) ([]float32, error) {
	if len(rgbHWC) != InputSize*InputSize*3 {
		return nil, fmt.Errorf("input must be %d bytes (got %d)", InputSize*InputSize*3, len(rgbHWC))
	}

	// rembg quirk worth matching: it divides the pixel array by max(im.max(), 1e-6),
	// NOT by 255. For a typical photo that's identical (255/255 → 1.0); for
	// underexposed crops it stretches before mean/std. Either way we mirror
	// it to stay on-parity with the rembg cutouts we proved.
	var pxMax uint8
	for _, v := range rgbHWC {
		if v > pxMax {
			pxMax = v
		}
	}
	maxF := float32(pxMax)
	if maxF < 1e-6 {
		maxF = 1e-6
	}

	data := s.inputTensor.GetData()
	// CHW layout: data[c*H*W + y*W + x]
	const HW = InputSize * InputSize
	for y := 0; y < InputSize; y++ {
		row := y * InputSize * 3
		for x := 0; x < InputSize; x++ {
			idx := row + x*3
			r := float32(rgbHWC[idx]) / maxF
			g := float32(rgbHWC[idx+1]) / maxF
			b := float32(rgbHWC[idx+2]) / maxF
			data[0*HW+y*InputSize+x] = (r - mean[0]) / std[0]
			data[1*HW+y*InputSize+x] = (g - mean[1]) / std[1]
			data[2*HW+y*InputSize+x] = (b - mean[2]) / std[2]
		}
	}

	if err := s.inner.Run(); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	raw := s.outputTens.GetData()
	mask := make([]float32, HW)
	// Sigmoid + min/max normalise → [0, 1].
	var mn, mx float32 = math.MaxFloat32, -math.MaxFloat32
	for i := 0; i < HW; i++ {
		v := float32(1.0 / (1.0 + math.Exp(float64(-raw[i]))))
		mask[i] = v
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	span := mx - mn
	if span < 1e-6 {
		// Flat mask — return as-is; the caller can decide what to do.
		return mask, nil
	}
	for i := range mask {
		mask[i] = (mask[i] - mn) / span
	}
	return mask, nil
}
