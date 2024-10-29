package lib

/*
#cgo LDFLAGS: -L/usr/local/lib -L/usr/local/cuda/lib64 -lcryptocuda -lblst -lcuda -lcudart -lm -lstdc++ -lgomp
#cgo CFLAGS: -I../../../native -I/usr/local/cuda/include -DFEATURE_BN254 -D__ADX__ -fopenmp
#include "lib.h"
#include "msm/msm.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func BoolToInt8(b bool) int8 {
	if b {
		return 1
	}
	return 0
}

func toCNTTConfig(cfg NTTConfig) C.NTT_Config {
	return C.NTT_Config{
		batches:               C.uint(cfg.Batches),
		order:                 C.NTT_InputOutputOrder(cfg.Order),
		ntt_type:              C.NTT_Type(cfg.NttType),
		extension_rate_bits:   C.uint(cfg.ExtensionRateBits),
		are_inputs_on_device:  C.char(BoolToInt8(cfg.AreInputsOnDevice)),
		are_outputs_on_device: C.char(BoolToInt8(cfg.AreOutputsOnDevice)),
		with_coset:            C.char(BoolToInt8(cfg.WithCoset)),
		is_multi_gpu:          C.char(BoolToInt8(cfg.IsMultiGPU)),
		is_coeffs:             C.char(BoolToInt8(cfg.IsCoeffs)),
		salt_size:             C.uint(cfg.SaltSize),
	}
}

func toCNTTTransposeConfig(cfg TransposeConfig) C.NTT_TransposeConfig {
	return C.NTT_TransposeConfig{
		batches:               C.uint(cfg.Batches),
		are_inputs_on_device:  C.char(BoolToInt8(cfg.AreInputsOnDevice)),
		are_outputs_on_device: C.char(BoolToInt8(cfg.AreOutputsOnDevice)),
	}
}

func ListDevicesInfo() error {
	err := C.list_devices_info()
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func GetNumberOfGPUs() (int, error) {
	var nums C.size_t
	err := C.get_number_of_gpus(&nums)
	if err.code != 0 {
		return 0, fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return int(nums), nil
}

func InitTwiddleFactors(deviceID, lgN int) error {
	err := C.init_twiddle_factors(C.size_t(deviceID), C.size_t(lgN))
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func InitCoset(deviceID, lgN int, cosetGen uint64) error {
	err := C.init_coset(C.size_t(deviceID), C.size_t(lgN), C.uint64_t(cosetGen))
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func InitCUDA() {
	C.init_cuda()
}

func InitCUDADegree(maxDegree int) {
	C.init_cuda_degree(C.uint(maxDegree))
}

func LDEBatch(deviceID int, output, input unsafe.Pointer, logNSize int, cfg NTTConfig) error {
	ccfg := toCNTTConfig(cfg)
	err := C.compute_batched_lde(
		C.size_t(deviceID),
		output,
		input,
		C.uint(logNSize),
		C.NTT_Direction(C.forward),
		ccfg,
	)
	if err.code != 0 {
		return fmt.Errorf("error: (code %d) %s", err.code, C.GoString(err.message))
	}
	return nil
}

func LDEBatchMultiGPU(output, input unsafe.Pointer, numGPU int, cfg NTTConfig, logNSize int) error {
	ccfg := toCNTTConfig(cfg)
	err := C.compute_batched_lde_multi_gpu(
		output,
		input,
		C.uint(numGPU),
		C.NTT_Direction(C.forward),
		ccfg,
		C.uint(logNSize),
	)
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func NTTBatch(deviceID int, inout unsafe.Pointer, logNSize int, cfg NTTConfig) error {
	ccfg := toCNTTConfig(cfg)
	err := C.compute_batched_ntt(
		C.size_t(deviceID),
		inout,
		C.uint(logNSize),
		C.NTT_Direction(C.forward),
		ccfg,
	)
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func INTTBatch(deviceID int, inout unsafe.Pointer, logNSize int, cfg NTTConfig) error {
	ccfg := toCNTTConfig(cfg)
	err := C.compute_batched_ntt(
		C.size_t(deviceID),
		inout,
		C.uint(logNSize),
		C.NTT_Direction(C.inverse),
		ccfg,
	)
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func TransposeRevBatch(deviceID int, output, input unsafe.Pointer, logNSize int, cfg TransposeConfig) error {
	ccfg := toCNTTTransposeConfig(cfg)
	err := C.compute_transpose_rev(
		C.ulong(deviceID),
		output,
		input,
		C.uint(logNSize),
		ccfg,
	)
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func FillDigestsBufLinearGPUWithGPUPtr(deviceID int, gpu_digests, gpu_caps, gpu_leaves unsafe.Pointer, nDigests, nCaps, nLeaves, leafSize, capH, hashType int) error {
	C.fill_digests_buf_linear_gpu_with_gpu_ptr(
		gpu_digests,
		gpu_caps,
		gpu_leaves,
		C.uint64_t(nDigests),
		C.uint64_t(nCaps),
		C.uint64_t(nLeaves),
		C.uint64_t(leafSize),
		C.uint64_t(capH),
		C.uint64_t(hashType),
		C.uint64_t(deviceID))

	return nil
}

func FillDigestsBufLinearMultiGPUWithGPUPtr(gpu_digests, gpu_caps, gpu_leaves unsafe.Pointer, nDigests, nCaps, nLeaves, leafSize, capH, hashType int) error {
	C.fill_digests_buf_linear_multigpu_with_gpu_ptr(
		gpu_digests,
		gpu_caps,
		gpu_leaves,
		C.uint64_t(nDigests),
		C.uint64_t(nCaps),
		C.uint64_t(nLeaves),
		C.uint64_t(leafSize),
		C.uint64_t(capH),
		C.uint64_t(hashType))

	return nil
}

func FillDigestsBufLinearCPU(cpu_digests, cpu_caps, cpu_leaves unsafe.Pointer, nDigests, nCaps, nLeaves, leafSize, capH, hashType int) error {
	C.fill_digests_buf_linear_cpu(
		cpu_digests,
		cpu_caps,
		cpu_leaves,
		C.uint64_t(nDigests),
		C.uint64_t(nCaps),
		C.uint64_t(nLeaves),
		C.uint64_t(leafSize),
		C.uint64_t(capH),
		C.uint64_t(hashType))

	return nil
}
