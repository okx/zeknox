package device

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type HostOrDeviceSlice[T any] struct {
	host   []T
	device struct {
		id int
		s  []T
	}
	isDevice bool
}

func (h *HostOrDeviceSlice[T]) Len() int {
	if h.isDevice {
		return len(h.device.s)
	}
	return len(h.host)
}

func (h *HostOrDeviceSlice[T]) IsEmpty() bool {
	return h.Len() == 0
}

func (h *HostOrDeviceSlice[T]) IsOnDevice() bool {
	return h.isDevice
}

func (h *HostOrDeviceSlice[T]) AsSlice() []T {
	if h.isDevice {
		panic("Use CopyToHost to move device data to a slice")
	}
	return h.host
}

func (h *HostOrDeviceSlice[T]) AsPtr() unsafe.Pointer {
	if h.isDevice {
		return unsafe.Pointer(&h.device.s[0])
	}
	return unsafe.Pointer(&h.host[0])
}

func (h *HostOrDeviceSlice[T]) OnHost(src []T) {
	h.host = src
	h.isDevice = false
}

func NewEmpty[T any]() *HostOrDeviceSlice[T] {
	return &HostOrDeviceSlice[T]{
		host:     nil,
		isDevice: false,
	}
}

func CudaMalloc[T any](deviceID int, count int) (*HostOrDeviceSlice[T], error) {
	size := count * int(unsafe.Sizeof(*new(T)))
	if size == 0 {
		return nil, errors.New("cudaErrorMemoryAllocation")
	}

	var devicePtr unsafe.Pointer
	C.cudaSetDevice(C.int(deviceID))
	var freeMem, totalMem C.size_t
	C.cudaMemGetInfo(&freeMem, &totalMem)
	if size > int(freeMem) {
		fmt.Printf("WARNING: not enough free GPU memory (needed: %d B, available: %d B, total %d B)!\n", size, freeMem, totalMem)
	}
	if err := C.cudaMalloc(&devicePtr, C.size_t(size)); err != 0 {
		return nil, errors.New("cudaErrorMemoryAllocation")
	}

	return &HostOrDeviceSlice[T]{
		device: struct {
			id int
			s  []T
		}{
			id: deviceID,
			s:  unsafe.Slice((*T)(devicePtr), count),
		},
		isDevice: true,
	}, nil
}

func (h *HostOrDeviceSlice[T]) CopyFromHost(val []T) error {
	if !h.isDevice {
		panic("Need device memory to copy into, and not host")
	}
	if len(h.device.s) != len(val) {
		return errors.New("destination and source slices have different lengths")
	}
	size := len(val) * int(unsafe.Sizeof(*new(T)))
	if size != 0 {
		C.cudaSetDevice(C.int(h.device.id))
		if err := C.cudaMemcpy(h.AsPtr(), unsafe.Pointer(&val[0]), C.size_t(size), C.cudaMemcpyHostToDevice); err != 0 {
			return errors.New("cudaMemcpy failed")
		}
	}
	return nil
}

func (h *HostOrDeviceSlice[T]) CopyToHost(val []T) error {
	if !h.isDevice {
		panic("Need device memory to copy from, and not host")
	}
	if len(h.device.s) != len(val) {
		return errors.New("destination and source slices have different lengths")
	}
	size := len(val) * int(unsafe.Sizeof(*new(T)))
	if size != 0 {
		C.cudaSetDevice(C.int(h.device.id))
		if err := C.cudaMemcpy(unsafe.Pointer(&val[0]), h.AsPtr(), C.size_t(size), C.cudaMemcpyDeviceToHost); err != 0 {
			return errors.New("cudaMemcpy failed")
		}
	}
	return nil
}

func (h *HostOrDeviceSlice[T]) Free() {
	if h.isDevice && len(h.device.s) > 0 {
		C.cudaSetDevice(C.int(h.device.id))
		C.cudaFree(h.AsPtr())
	}
}
