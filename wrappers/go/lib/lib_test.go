package lib

import (
	"math"
	"testing"
	"unsafe"
	"zeknox/device"
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

func TestMerkleTreeBuildingOutputGpu(t *testing.T) {
	leaf := []uint64{8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027}

	expHash := []uint64{7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017}

	leaf_size := len(leaf)
	n_leaves := 4
	n_caps := n_leaves
	hash_size := 4
	n_digests := 2 * (n_leaves - n_caps)
	cap_h := int(math.Log2(float64(n_caps)))

	leaves := make([]uint64, n_leaves*leaf_size)
	for i := 0; i < n_leaves; i++ {
		copy(leaves[i*leaf_size:], leaf)
	}
	caps := make([]uint64, n_caps*hash_size)

	gpuLeavesBuff, _ := device.CudaMalloc[uint64](0, n_leaves*leaf_size)
	gpuCapsBuff, _ := device.CudaMalloc[uint64](0, n_caps*hash_size)
	gpuDigestsBuff, _ := device.CudaMalloc[uint64](0, n_caps*hash_size)

	gpuLeavesBuff.CopyFromHost(leaves)

	FillDigestsBufLinearGPUWithGPUPtr(0, gpuDigestsBuff.AsPtr(), gpuCapsBuff.AsPtr(), gpuLeavesBuff.AsPtr(), n_digests, n_caps, n_leaves, leaf_size, cap_h, HashPoseidon)

	gpuCapsBuff.CopyToHost(caps)
	for j := 0; j < n_caps; j++ {
		for i := 0; i < len(expHash); i++ {
			if caps[4*j+i] != expHash[i] {
				t.Errorf("MerkleTreeBuildingOutput() = %v, want %v", caps, expHash)
			}
		}
	}
}

func TestMerkleTreeBuildingOutputMultiGpu(t *testing.T) {
	num, err := GetNumberOfGPUs()
	if err != nil {
		t.Errorf("GetNumberOfGPUs() error = %v", err)
	}
	if num < 2 {
		t.Skip("not enough GPUs to test LDEBatchMultiGPU")
	}

	leaf := []uint64{8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027}

	expHash := []uint64{7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017}

	leaf_size := len(leaf)
	n_leaves := 4
	n_caps := n_leaves
	hash_size := 4
	n_digests := 2 * (n_leaves - n_caps)
	cap_h := int(math.Log2(float64(n_caps)))

	leaves := make([]uint64, n_leaves*leaf_size)
	for i := 0; i < n_leaves; i++ {
		copy(leaves[i*leaf_size:], leaf)
	}
	caps := make([]uint64, n_caps*hash_size)

	gpuLeavesBuff, _ := device.CudaMalloc[uint64](0, n_leaves*leaf_size)
	gpuCapsBuff, _ := device.CudaMalloc[uint64](0, n_caps*hash_size)
	gpuDigestsBuff, _ := device.CudaMalloc[uint64](0, n_caps*hash_size)

	gpuLeavesBuff.CopyFromHost(leaves)

	FillDigestsBufLinearMultiGPUWithGPUPtr(gpuDigestsBuff.AsPtr(), gpuCapsBuff.AsPtr(), gpuLeavesBuff.AsPtr(), n_digests, n_caps, n_leaves, leaf_size, cap_h, HashPoseidon)

	gpuCapsBuff.CopyToHost(caps)
	for j := 0; j < n_caps; j++ {
		for i := 0; i < len(expHash); i++ {
			if caps[4*j+i] != expHash[i] {
				t.Errorf("MerkleTreeBuildingOutput() = %v, want %v", caps, expHash)
			}
		}
	}
}

func TestMerkleTreeBuildingOutputCpu(t *testing.T) {
	leaf := []uint64{8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027}

	expHash := []uint64{7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017}

	leaf_size := len(leaf)
	n_leaves := 4
	n_caps := n_leaves
	hash_size := 4
	n_digests := 2 * (n_leaves - n_caps)
	cap_h := int(math.Log2(float64(n_caps)))

	leaves := make([]uint64, n_leaves*leaf_size)
	for i := 0; i < n_leaves; i++ {
		copy(leaves[i*leaf_size:], leaf)
	}
	caps := make([]uint64, n_caps*hash_size)

	FillDigestsBufLinearCPU(unsafe.Pointer(&caps[0]), unsafe.Pointer(&caps[0]), unsafe.Pointer(&leaves[0]), n_digests, n_caps, n_leaves, leaf_size, cap_h, HashPoseidon)

	for j := 0; j < n_caps; j++ {
		for i := 0; i < len(expHash); i++ {
			if caps[4*j+i] != expHash[i] {
				t.Errorf("MerkleTreeBuildingOutput() = %v, want %v", caps, expHash)
			}
		}
	}
}
