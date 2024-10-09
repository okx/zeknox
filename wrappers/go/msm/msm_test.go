package msm

import (
	"fmt"
	"math/big"
	"testing"
	"unsafe"
	"zeknox/device"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"
)

func TestMsmG1InputsNotOnDevice(t *testing.T) {
	var npoints uint32 = 1024
	point_scale_factor := 10
	scalars := make([]fr.Element, npoints)
	for i := 0; i < len(scalars); i++ {
		scalars[i] = fr.NewElement(uint64(i + 1))
	}

	_, _, g1GenAff, _ := curve.Generators()
	basePointAffine := curve.G1Affine{}
	basePointAffine.ScalarMultiplication(&g1GenAff, big.NewInt(int64(point_scale_factor)))

	points := make([]curve.G1Affine, npoints)
	for i := 0; i < len(points); i++ {
		points[i] = basePointAffine
	}

	fmt.Printf("g1GenAff %s \n", g1GenAff.String())
	fmt.Printf("basePointAffine %s \n", basePointAffine.String())

	cpuResultAffine := curve.G1Affine{}
	_, err := cpuResultAffine.MultiExp(points, scalars, ecc.MultiExpConfig{NbTasks: 16})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}
	fmt.Printf("cpuResultAffine %s \n", cpuResultAffine.String())

	expectedResultAffine := curve.G1Affine{}
	expectedResultAffine.ScalarMultiplication(&g1GenAff, big.NewInt(int64((1+npoints)*npoints/2)*int64(point_scale_factor)))
	assert.True(t, cpuResultAffine.Equal(&expectedResultAffine), "cpu result is incorrect")

	gpuResultJac := curve.G1Jac{}

	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = false
	cfg.ArePointsInMont = true
	cfg.Npoints = npoints
	cfg.FfiAffineSz = 64
	err = MSM_G1(unsafe.Pointer(&gpuResultJac), unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), 0, cfg)
	if err != nil {
		t.Errorf("invoke msm gpu error: %s", err.Error())
	}
	gpuResultAffine := curve.G1Affine{}
	gpuResultAffine.FromJacobian(&gpuResultJac)
	fmt.Printf("gpuResultAffine: %s \n", gpuResultAffine.String())
	assert.True(t, cpuResultAffine.Equal(&gpuResultAffine), "gpu result is incorrect")

}

func TestMsmG1InputsOnDevice(t *testing.T) {

	var npoints uint32 = 1024
	point_scale_factor := 10
	scalars := make([]fr.Element, npoints)
	for i := 0; i < len(scalars); i++ {
		scalars[i] = fr.NewElement(uint64(i + 1))
	}

	_, _, g1GenAff, _ := curve.Generators()
	basePointAffine := curve.G1Affine{}
	basePointAffine.ScalarMultiplication(&g1GenAff, big.NewInt(int64(point_scale_factor)))

	points := make([]curve.G1Affine, npoints)
	for i := 0; i < len(points); i++ {
		points[i] = basePointAffine
	}

	fmt.Printf("g1GenAff %s \n", g1GenAff.String())
	fmt.Printf("basePointAffine %s \n", basePointAffine.String())

	cpuResultAffine := curve.G1Affine{}
	_, err := cpuResultAffine.MultiExp(points, scalars, ecc.MultiExpConfig{NbTasks: 16})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}
	fmt.Printf("cpuResultAffine %s \n", cpuResultAffine.String())

	expectedResultAffine := curve.G1Affine{}
	expectedResultAffine.ScalarMultiplication(&g1GenAff, big.NewInt(int64((1+npoints)*npoints/2)*int64(point_scale_factor)))
	assert.True(t, cpuResultAffine.Equal(&expectedResultAffine), "cpu result is incorrect")

	d_scalar, err := device.CudaMalloc[fr.Element](0, int(npoints))
	if err != nil {
		t.Errorf("allocate scalars to device error: %s", err.Error())
	}
	err = d_scalar.CopyFromHost(scalars)
	if err != nil {
		t.Errorf("copy scalars to device error: %s", err.Error())
	}

	d_points, err := device.CudaMalloc[curve.G1Affine](0, int(npoints))
	if err != nil {
		t.Errorf("allocate points to device error: %s", err.Error())
	}
	err = d_points.CopyFromHost(points)
	if err != nil {
		t.Errorf("copy points to device error: %s", err.Error())
	}

	gpuResultJac := curve.G1Jac{}

	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = true
	cfg.ArePointsInMont = true
	cfg.Npoints = npoints
	cfg.FfiAffineSz = 64
	err = MSM_G1(unsafe.Pointer(&gpuResultJac), d_points.AsPtr(), d_scalar.AsPtr(), 0, cfg)
	if err != nil {
		t.Errorf("invoke msm gpu error: %s", err.Error())
	}
	gpuResultAffine := curve.G1Affine{}
	gpuResultAffine.FromJacobian(&gpuResultJac)
	fmt.Printf("gpuResultAffine: %s \n", gpuResultAffine.String())
	assert.True(t, cpuResultAffine.Equal(&gpuResultAffine), "gpu result is incorrect")

}
