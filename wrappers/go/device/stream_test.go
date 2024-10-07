package device

import (
	"testing"
)

func TestNewCudaStream(t *testing.T) {
	stream, err := NewCudaStream()
	if err != nil {
		t.Fatalf("Failed to create new CUDA stream: %v", err)
	}
	defer stream.Destroy()

	if stream.IsNull() {
		t.Fatalf("Expected non-null CUDA stream handle")
	}
}

func TestNewCudaStreamWithFlags(t *testing.T) {
	stream, err := NewCudaStreamWithFlags(CudaStreamNonBlocking)
	if err != nil {
		t.Fatalf("Failed to create new CUDA stream with flags: %v", err)
	}
	defer stream.Destroy()

	if stream.IsNull() {
		t.Fatalf("Expected non-null CUDA stream handle")
	}
}

func TestCudaStreamDestroy(t *testing.T) {
	stream, err := NewCudaStream()
	if err != nil {
		t.Fatalf("Failed to create new CUDA stream: %v", err)
	}

	err = stream.Destroy()
	if err != nil {
		t.Fatalf("Failed to destroy CUDA stream: %v", err)
	}

	if !stream.IsNull() {
		t.Fatalf("Expected null CUDA stream handle after destroy")
	}
}

func TestCudaStreamSynchronize(t *testing.T) {
	stream, err := NewCudaStream()
	if err != nil {
		t.Fatalf("Failed to create new CUDA stream: %v", err)
	}
	defer stream.Destroy()

	err = stream.Synchronize()
	if err != nil {
		t.Fatalf("Failed to synchronize CUDA stream: %v", err)
	}
}

func TestCudaStreamSetHandle(t *testing.T) {
	stream1, err := NewCudaStream()
	if err != nil {
		t.Fatalf("Failed to create new CUDA stream: %v", err)
	}

	stream2, err := NewCudaStream()
	if err != nil {
		t.Fatalf("Failed to create new CUDA stream: %v", err)
	}

	stream1.SetHandle(stream2.Handle())
	if stream1.Handle() != stream2.Handle() {
		t.Fatalf("Expected CUDA stream handle to be updated!")
	}

	stream1.Destroy()
}
