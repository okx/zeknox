// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

package lib

import (
	"math"
	"testing"
	"unsafe"

	"github.com/okx/zeknox/wrappers/go/device"
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
	rateBits := 2
	logDomainSize := logn + rateBits
	batches := 2

	InitTwiddleFactors(0, logDomainSize)
	InitCoset(0, logDomainSize, 7)

	a := make([]uint64, batches*(1<<logn))
	for i := 0; i < batches*(1<<logn); i++ {
		a[i] = uint64(i)
	}
	b := make([]uint64, batches*(1<<logDomainSize))
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])

	cfg := DefaultNTTConfig()
	cfg.ExtensionRateBits = uint32(rateBits)
	err := LDEBatch(0, pb, pa, logDomainSize, cfg)
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
	rateBits := 2
	logDomainSize := logn + rateBits
	batches := 2

	for i := 0; i < num; i++ {
		InitTwiddleFactors(i, logDomainSize)
		InitCoset(i, logDomainSize, 7)
	}

	a := make([]uint64, batches*(1<<logn))
	for i := 0; i < batches*(1<<logn); i++ {
		a[i] = uint64(i)
	}
	b := make([]uint64, batches*(1<<logDomainSize))
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])

	cfg := DefaultNTTConfig()
	cfg.ExtensionRateBits = uint32(rateBits)
	err = LDEBatchMultiGPU(pb, pa, num, cfg, logDomainSize, batches*(1<<logn), batches*(1<<logDomainSize))
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

func caseTestMerkleTreeBuildingOutputGpu(t *testing.T, nLeaves int, nCaps int, leaf, expHash []uint64) {
	leafSize := len(leaf)
	hashSize := 4
	nDigests := 2 * (nLeaves - nCaps)
	capHeight := int(math.Log2(float64(nCaps)))
	nDigestsAlloc := nDigests
	if nDigests == 0 {
		nDigestsAlloc = 1
	}

	leaves := make([]uint64, nLeaves*leafSize)
	for i := 0; i < nLeaves; i++ {
		copy(leaves[i*leafSize:], leaf)
	}
	caps := make([]uint64, nCaps*hashSize)

	gpuLeavesBuff, _ := device.CudaMalloc[uint64](0, nLeaves*leafSize)
	gpuCapsBuff, _ := device.CudaMalloc[uint64](0, nCaps*hashSize)
	gpuDigestsBuff, _ := device.CudaMalloc[uint64](0, nDigestsAlloc*hashSize)

	gpuLeavesBuff.CopyFromHost(leaves)

	FillDigestsBufLinearGPUWithGPUPtr(0, gpuDigestsBuff.AsPtr(), gpuCapsBuff.AsPtr(), gpuLeavesBuff.AsPtr(), nDigests, nCaps, nLeaves, leafSize, capHeight, HashPoseidon)

	gpuCapsBuff.CopyToHost(caps)
	for j := 0; j < nCaps; j++ {
		for i := 0; i < len(expHash); i++ {
			if caps[4*j+i] != expHash[i] {
				t.Errorf("MerkleTreeBuildingOutput() = %v, want %v", caps, expHash)
				return
			}
		}
	}
}

func TestMerkleTreeBuildingOutputGpu(t *testing.T) {
	leaf := []uint64{8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027}

	// case 1: nCaps = nLeaves
	nLeaves := 8
	nCaps := nLeaves
	expHash := []uint64{7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017}
	caseTestMerkleTreeBuildingOutputGpu(t, nLeaves, nCaps, leaf, expHash)

	// case 2: subtree_leaves_len = 2
	nLeaves = 8
	nCaps = 4
	expHash = []uint64{7320113784629306061, 6779895905614599055, 11363085033710007978, 10376477785629036773}
	caseTestMerkleTreeBuildingOutputGpu(t, nLeaves, nCaps, leaf, expHash)

	// case 3:
	nLeaves = 16
	nCaps = 2
	expHash = []uint64{15643713602922591410, 15764162833178839023, 7963799186968795376, 10933634831850922421}
	caseTestMerkleTreeBuildingOutputGpu(t, nLeaves, nCaps, leaf, expHash)
}

func caseTestMerkleTreeBuildingOutputMultiGpu(t *testing.T, nLeaves int, nCaps int, leaf, expHash []uint64) {
	hashSize := 4
	leafSize := len(leaf)
	nDigests := 2 * (nLeaves - nCaps)
	capHeight := int(math.Log2(float64(nCaps)))
	nDigestsAlloc := nDigests
	if nDigests == 0 {
		nDigestsAlloc = 1
	}

	leaves := make([]uint64, nLeaves*leafSize)
	for i := 0; i < nLeaves; i++ {
		copy(leaves[i*leafSize:], leaf)
	}
	caps := make([]uint64, nCaps*hashSize)

	gpuLeavesBuff, _ := device.CudaMalloc[uint64](0, nLeaves*leafSize)
	gpuCapsBuff, _ := device.CudaMalloc[uint64](0, nCaps*hashSize)
	gpuDigestsBuff, _ := device.CudaMalloc[uint64](0, nDigestsAlloc*hashSize)

	gpuLeavesBuff.CopyFromHost(leaves)

	FillDigestsBufLinearMultiGPUWithGPUPtr(gpuDigestsBuff.AsPtr(), gpuCapsBuff.AsPtr(), gpuLeavesBuff.AsPtr(), nDigests, nCaps, nLeaves, leafSize, capHeight, HashPoseidon)

	gpuCapsBuff.CopyToHost(caps)
	for j := 0; j < nCaps; j++ {
		for i := 0; i < len(expHash); i++ {
			if caps[4*j+i] != expHash[i] {
				t.Errorf("MerkleTreeBuildingOutput() = %v, want %v", caps, expHash)
				return
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

	// case 1: nCaps = nLeaves
	nLeaves := num
	nCaps := nLeaves
	expHash := []uint64{7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017}
	caseTestMerkleTreeBuildingOutputMultiGpu(t, nLeaves, nCaps, leaf, expHash)

	// case 2: subtree_leaves_len = 2
	nLeaves = 2 * num
	nCaps = num
	expHash = []uint64{7320113784629306061, 6779895905614599055, 11363085033710007978, 10376477785629036773}
	caseTestMerkleTreeBuildingOutputMultiGpu(t, nLeaves, nCaps, leaf, expHash)

	// case 3: 1 subtree per GPU
	nLeaves = 16 * num
	nCaps = 2 * num
	expHash = []uint64{15643713602922591410, 15764162833178839023, 7963799186968795376, 10933634831850922421}
	caseTestMerkleTreeBuildingOutputMultiGpu(t, nLeaves, nCaps, leaf, expHash)
}

func caseTestMerkleTreeBuildingOutputCpu(t *testing.T, nLeaves int, nCaps int, leaf, expHash []uint64) {
	leafSize := len(leaf)
	hashSize := 4
	nDigests := 2 * (nLeaves - nCaps)
	capHeight := int(math.Log2(float64(nCaps)))
	nDigestsAlloc := nDigests
	if nDigests == 0 {
		nDigestsAlloc = 1
	}

	leaves := make([]uint64, nLeaves*leafSize)
	for i := 0; i < nLeaves; i++ {
		copy(leaves[i*leafSize:], leaf)
	}
	caps := make([]uint64, nCaps*hashSize)
	digests := make([]uint64, nDigestsAlloc*hashSize)

	FillDigestsBufLinearCPU(unsafe.Pointer(&digests[0]), unsafe.Pointer(&caps[0]), unsafe.Pointer(&leaves[0]), nDigests, nCaps, nLeaves, leafSize, capHeight, HashPoseidon)

	for j := 0; j < nCaps; j++ {
		for i := 0; i < len(expHash); i++ {
			if caps[4*j+i] != expHash[i] {
				t.Errorf("MerkleTreeBuildingOutput() = %v, want %v", caps, expHash)
				return
			}
		}
	}
}

func TestMerkleTreeBuildingOutputCpu(t *testing.T) {
	leaf := []uint64{8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027}

	// case 1: nCaps = nLeaves
	nLeaves := 8
	nCaps := nLeaves
	expHash := []uint64{7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017}
	caseTestMerkleTreeBuildingOutputCpu(t, nLeaves, nCaps, leaf, expHash)

	// case 2: subtree_leaves_len = 2
	nLeaves = 8
	nCaps = 4
	expHash = []uint64{7320113784629306061, 6779895905614599055, 11363085033710007978, 10376477785629036773}
	caseTestMerkleTreeBuildingOutputCpu(t, nLeaves, nCaps, leaf, expHash)

	// case 3:
	nLeaves = 16
	nCaps = 2
	expHash = []uint64{15643713602922591410, 15764162833178839023, 7963799186968795376, 10933634831850922421}
	caseTestMerkleTreeBuildingOutputCpu(t, nLeaves, nCaps, leaf, expHash)
}
