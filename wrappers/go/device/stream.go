package device

/*
#cgo LDFLAGS: -lcuda
#include <cuda_runtime.h>
*/
import "C"
import (
	"fmt"
)

type CudaStream struct {
	handle C.cudaStream_t
}

type CudaStreamCreateFlags uint32

const (
	CudaStreamDefault     CudaStreamCreateFlags = C.cudaStreamDefault
	CudaStreamNonBlocking CudaStreamCreateFlags = C.cudaStreamNonBlocking
)

func NewCudaStream() (*CudaStream, error) {
	var handle C.cudaStream_t
	err := result(C.cudaStreamCreate(&handle))
	if err != nil {
		return nil, err
	}
	return &CudaStream{handle: handle}, nil
}

func NewCudaStreamWithFlags(flags CudaStreamCreateFlags) (*CudaStream, error) {
	var handle C.cudaStream_t
	err := result(C.cudaStreamCreateWithFlags(&handle, C.uint(flags)))
	if err != nil {
		return nil, err
	}
	return &CudaStream{handle: handle}, nil
}

func (s *CudaStream) Destroy() error {
	if s.handle == nil {
		return nil
	}
	err := result(C.cudaStreamDestroy(s.handle))
	s.handle = nil
	return err
}

func (s *CudaStream) Synchronize() error {
	return result(C.cudaStreamSynchronize(s.handle))
}

func result(code C.cudaError_t) error {
	if code != C.cudaSuccess {
		return fmt.Errorf("CUDA error: %s", C.GoString(C.cudaGetErrorString(code)))
	}
	return nil
}

func (s *CudaStream) Handle() C.cudaStream_t {
	return s.handle
}

func (s *CudaStream) IsNull() bool {
	return s.handle == nil
}

func (s *CudaStream) SetHandle(handle C.cudaStream_t) {
	s.handle = handle
}
