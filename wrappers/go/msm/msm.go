package msm

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

	"github.com/okx/cryptography_cuda/wrappers/go/lib"
)

type MSMConfig struct {
	FfiAffineSz        uint32
	Npoints            uint32
	ArePointsInMont    bool
	AreInputsOnDevice  bool
	AreOutputsOnDevice bool
	LargeBucketFactor  uint32
	BigTriangle        bool
}

func DefaultMSMConfig() MSMConfig {
	return MSMConfig{
		FfiAffineSz:        0,
		Npoints:            0,
		ArePointsInMont:    false,
		AreInputsOnDevice:  false,
		AreOutputsOnDevice: false,
		LargeBucketFactor:  0,
		BigTriangle:        false,
	}
}

func toCMSMConfig(cfg MSMConfig) C.MSM_Config {
	return C.MSM_Config{
		ffi_affine_sz:         C.uint(cfg.FfiAffineSz),
		npoints:               C.uint(cfg.Npoints),
		are_points_in_mont:    C.char(lib.BoolToInt8(cfg.ArePointsInMont)),
		are_inputs_on_device:  C.char(lib.BoolToInt8(cfg.AreInputsOnDevice)),
		are_outputs_on_device: C.char(lib.BoolToInt8(cfg.AreOutputsOnDevice)),
		large_bucket_factor:   C.uint(cfg.LargeBucketFactor),
		big_triangle:          C.char(lib.BoolToInt8(cfg.BigTriangle)),
	}
}

func MSM_G1(output, input_points, input_scalars unsafe.Pointer, numGPU int, cfg MSMConfig) error {
	ccfg := toCMSMConfig(cfg)
	// fmt.Printf("start invoke mul pippenger %v\n", ccfg)
	err := C.mult_pippenger(
		C.uint(numGPU),
		output,
		input_points,
		input_scalars,
		ccfg,
	)
	C.fflush(C.stdout)
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func MSM_G2(output, input_points, input_scalars unsafe.Pointer, numGPU int, cfg MSMConfig) error {
	ccfg := toCMSMConfig(cfg)
	// fmt.Printf("start invoke mult_pippenger_g2, npoints: %d\n", ccfg.npoints)
	err := C.mult_pippenger_g2(
		C.uint(numGPU),
		output,
		input_points,
		input_scalars,
		ccfg,
	)
	C.fflush(C.stdout)
	if err.code != 0 {
		return fmt.Errorf("error: %s", C.GoString(err.message))
	}
	return nil
}

func Alloc_c(size int) unsafe.Pointer {
	data := C.malloc(C.size_t(size))
	return data
}
