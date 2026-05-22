package session

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/Aayush9029/peel/internal/config"
)

const InputSize = 1024

const (
	inputName  = "input_image"
	outputName = "output_image"
)

// ImageNet normalisation — BiRefNet sits on a Swin backbone and was trained
// against these. Wrong values produce masks that look almost-right but read
// as washed-out alpha.
var (
	mean = [3]float32{0.485, 0.456, 0.406}
	std  = [3]float32{0.229, 0.224, 0.225}
)

var (
	envOnce sync.Once
	envErr  error
)

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
		}
	})
	return envErr
}

// Session is not safe for concurrent use.
type Session struct {
	inner       *ort.AdvancedSession
	inputTensor *ort.Tensor[float32]
	outputTens  *ort.Tensor[float32]
}

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

	return &Session{inner: inner, inputTensor: inputTensor, outputTens: outputTensor}, nil
}

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

// Predict expects exactly 1024*1024*3 RGB bytes and returns a 1024² alpha mask
// in [0,1] row-major. Pre/post-processing matches rembg's BiRefNet session.
func (s *Session) Predict(rgbHWC []uint8) ([]float32, error) {
	if len(rgbHWC) != InputSize*InputSize*3 {
		return nil, fmt.Errorf("input must be %d bytes (got %d)", InputSize*InputSize*3, len(rgbHWC))
	}

	// rembg parity: divide by max(pixel, 1e-6), NOT by 255. For most photos
	// these are equivalent; on underexposed crops it stretches before mean/std.
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
		return mask, nil
	}
	for i := range mask {
		mask[i] = (mask[i] - mn) / span
	}
	return mask, nil
}
