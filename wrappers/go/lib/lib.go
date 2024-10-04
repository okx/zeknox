package lib

/*
#cgo LDFLAGS: -L../../../native/build -lcryptocuda -lcuda -lcudart -lm -lstdc++
#cgo CFLAGS: -I../../../native -DFEATURE_GOLDILOCKS
#include <stdlib.h>
#include "lib.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func BoolToInt(b bool) int {
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
		are_inputs_on_device:  C.int(BoolToInt(cfg.AreInputsOnDevice)),
		are_outputs_on_device: C.int(BoolToInt(cfg.AreOutputsOnDevice)),
		with_coset:            C.int(BoolToInt(cfg.WithCoset)),
		is_multi_gpu:          C.int(BoolToInt(cfg.IsMultiGPU)),
		salt_size:             C.uint(cfg.SaltSize),
	}
}

func toCNTTTransposeConfig(cfg TransposeConfig) C.NTT_TransposeConfig {
	return C.NTT_TransposeConfig{
		batches:               C.uint(cfg.Batches),
		are_inputs_on_device:  C.int(BoolToInt(cfg.AreInputsOnDevice)),
		are_outputs_on_device: C.int(BoolToInt(cfg.AreOutputsOnDevice)),
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

func LDEBatchMultiGPU(output, input unsafe.Pointer, numGPU int, cfg NTTConfig, logNSize, totalNumInputElements, totalNumOutputElements int) error {
	ccfg := toCNTTConfig(cfg)
	err := C.compute_batched_lde_multi_gpu(
		output,
		input,
		C.uint(numGPU),
		C.NTT_Direction(C.forward),
		ccfg,
		C.uint(logNSize),
		C.size_t(totalNumInputElements),
		C.size_t(totalNumOutputElements),
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

func NaiveTransposeRevBatch(deviceID int, output, input unsafe.Pointer, logNSize int, cfg TransposeConfig) error {
	ccfg := toCNTTTransposeConfig(cfg)
	err := C.compute_naive_transpose_rev(
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
