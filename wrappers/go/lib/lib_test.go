package lib

import (
	"testing"
	"unsafe"
)

func TestListDevicesInfo(t *testing.T) {
	err := ListDevicesInfo()
	if err != nil {
		t.Errorf("ListDevicesInfo() error = %v", err)
	}
}

func TestGetNumberOfGPUs(t *testing.T) {
	num, err := GetNumberOfGPUs()
	if err != nil {
		t.Errorf("GetNumberOfGPUs() error = %v", err)
	}
	if num <= 0 {
		t.Errorf("GetNumberOfGPUs() = %v, want > 0", num)
	}
}

func TestInitTwiddleFactors(t *testing.T) {
	err := InitTwiddleFactors(0, 10)
	if err != nil {
		t.Errorf("InitTwiddleFactors() error = %v", err)
	}
}

func TestInitCoset(t *testing.T) {
	err := InitCoset(0, 10, 12345)
	if err != nil {
		t.Errorf("InitCoset() error = %v", err)
	}
}

func TestInitCUDA(t *testing.T) {
	InitCUDA()
	// No error expected, just ensure it doesn't panic
}

func TestInitCUDADegree(t *testing.T) {
	InitCUDADegree(10)
	// No error expected, just ensure it doesn't panic
}

func TestLDEBatch(t *testing.T) {
	logn := 4
	rate_bits := 2
	log_domain_size := logn + rate_bits
	batches := 2

	InitTwiddleFactors(0, log_domain_size)
	InitCoset(0, log_domain_size, 7)

	a := make([]uint64, batches*(1<<logn))
	for i := 0; i < batches*(1<<logn); i++ {
		a[i] = uint64(i)
	}
	b := make([]uint64, batches*(1<<log_domain_size))
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])

	cfg := DefaultNTTConfig()
	cfg.ExtensionRateBits = uint32(rate_bits)
	err := LDEBatch(0, pb, pa, log_domain_size, cfg)
	if err != nil {
		t.Errorf("LDEBatch() error = %v", err)
	}
}

func TestLDEBatchMultiGPU(t *testing.T) {
	num, err := GetNumberOfGPUs()
	if err != nil {
		t.Errorf("GetNumberOfGPUs() error = %v", err)
	}
	if num < 2 {
		t.Skip("not enough GPUs to test LDEBatchMultiGPU")
	}

	logn := 4
	rate_bits := 2
	log_domain_size := logn + rate_bits
	batches := 2

	for i := 0; i < num; i++ {
		InitTwiddleFactors(i, log_domain_size)
		InitCoset(i, log_domain_size, 7)
	}

	a := make([]uint64, batches*(1<<logn))
	for i := 0; i < batches*(1<<logn); i++ {
		a[i] = uint64(i)
	}
	b := make([]uint64, batches*(1<<log_domain_size))
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])

	cfg := DefaultNTTConfig()
	cfg.ExtensionRateBits = uint32(rate_bits)
	err = LDEBatchMultiGPU(pb, pa, num, cfg, log_domain_size, batches*(1<<logn), batches*(1<<log_domain_size))
	if err != nil {
		t.Errorf("LDEBatchMultiGPU() error = %v", err)
	}
}

func TestNTTBatch(t *testing.T) {
	logn := 10
	a := make([]uint64, 1<<logn)
	for i := 0; i < 1<<logn; i++ {
		a[i] = uint64(i)
	}
	cfg := DefaultNTTConfig()
	err := NTTBatch(0, unsafe.Pointer(&a[0]), logn, cfg)
	if err != nil {
		t.Errorf("NTTBatch() error = %v", err)
	}
}

func TestINTTBatch(t *testing.T) {
	logn := 10
	a := make([]uint64, 1<<logn)
	for i := 0; i < 1<<logn; i++ {
		a[i] = uint64(i)
	}
	cfg := DefaultNTTConfig()
	err := INTTBatch(0, unsafe.Pointer(&a[0]), logn, cfg)
	if err != nil {
		t.Errorf("INTTBatch() error = %v", err)
	}
}

func TestTransposeRevBatch(t *testing.T) {
	logn := 10
	size := 1 << logn
	a := make([]uint64, size)
	for i := 0; i < size; i++ {
		a[i] = uint64(i)
	}
	b := make([]uint64, size)
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])
	cfg := DefaultTransposeConfig()
	err := TransposeRevBatch(0, pb, pa, logn, cfg)
	if err != nil {
		t.Errorf("TransposeRevBatch() error = %v", err)
	}
}
