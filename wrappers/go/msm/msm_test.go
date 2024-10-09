package msm

import (
	"fmt"
	"testing"
	"unsafe"
	"zeknox/device"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestMsmG1(t *testing.T) {

	var device_alloc_size uint32 = 1024
	scalars := make([]fr.Element, device_alloc_size)
	for i := 0; i < len(scalars); i++ {
		scalars[i] = fr.NewElement(uint64(i))
	}

	// curve.G1Affine

	base := fp.NewElement(1)
	g := curve.MapToG1(base)
	points := make([]curve.G1Affine, device_alloc_size)
	for i := 0; i < len(points); i++ {
		points[i] = g
	}

	result, err := g.MultiExp(points, scalars, ecc.MultiExpConfig{NbTasks: 16})
	// expected := g.X.Mul()
	if err != nil {
		fmt.Printf("cpu msm error: %s", err.Error())
	}
	fmt.Printf("cpu result %s \n", result.String())

	d_scalar, err := device.CudaMalloc[fr.Element](0, int(device_alloc_size))
	if err != nil {
		fmt.Printf("allocate scalars to device error: %s", err.Error())
	}
	err = d_scalar.CopyFromHost(scalars)
	if err != nil {
		fmt.Printf("copy scalars to device error: %s", err.Error())
	}

	d_points, err := device.CudaMalloc[curve.G1Affine](0, int(device_alloc_size))
	if err != nil {
		fmt.Printf("allocate points to device error: %s", err.Error())
	}
	err = d_points.CopyFromHost(points)
	if err != nil {
		fmt.Printf("copy points to device error: %s", err.Error())
	}

	result_jac := curve.G1Jac{}
	// r := fp.NewElement(1)
	// result_affine := curve.MapToG1(r)
	// result_affine.to
	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = true
	cfg.Npoints = device_alloc_size
	cfg.FfiAffineSz = 64
	err = MSM_G1(unsafe.Pointer(&result_jac), d_points.AsPtr(), d_scalar.AsPtr(), 0, cfg)
	if err != nil {
		fmt.Printf("invoke msm gpu error: %s", err.Error())
	}
	fmt.Printf("result_jac %s", result_jac.String())

}
