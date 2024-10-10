package msm

import (
	"fmt"
	"math/big"
	"runtime"
	"testing"
	"unsafe"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"
)

func BenchmarkMsmG1InputsNotOnDevice(b *testing.B) {
	const npoints uint32 = 1 << 20
	var (
		points  [npoints]curve.G1Affine
		scalars [npoints]fr.Element
	)
	fillBenchScalars(scalars[:])
	fillBenchBasesG1(points[:])

	// CPU
	cpuNum := runtime.NumCPU()
	cpuResultAffine := curve.G1Affine{}
	b.Run(fmt.Sprintf("CPU %d", npoints), func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// NbTasks same as Gnark prover
			cpuResultAffine.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{NbTasks: cpuNum / 2})
		}
	})
	fmt.Printf("cpuResultAffine %s \n", cpuResultAffine.String())

	gpuResultJac := curve.G1Jac{}
	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = false
	cfg.ArePointsInMont = true
	cfg.Npoints = npoints
	cfg.FfiAffineSz = 64
	b.Run(fmt.Sprintf("GPU %d", npoints), func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MSM_G1(unsafe.Pointer(&gpuResultJac), unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), 0, cfg)
		}
	})
	gpuResultAffine := curve.G1Affine{}
	gpuResultAffine.FromJacobian(&gpuResultJac)
	fmt.Printf("gpuResultAffine: %s \n", gpuResultAffine.String())
	assert.True(b, cpuResultAffine.Equal(&gpuResultAffine), "gpu result is incorrect")
}

// func TestMsmG1InputsOnDevice(t *testing.T) {
// 	var npoints uint32 = 1 << 20
// 	point_scale_factor := 10
// 	scalars := make([]fr.Element, npoints)
// 	for i := 0; i < len(scalars); i++ {
// 		scalars[i] = fr.NewElement(uint64(i + 1))
// 	}

// 	_, _, g1GenAff, _ := curve.Generators()
// 	basePointAffine := curve.G1Affine{}
// 	basePointAffine.ScalarMultiplication(&g1GenAff, big.NewInt(int64(point_scale_factor)))

// 	points := make([]curve.G1Affine, npoints)
// 	for i := 0; i < len(points); i++ {
// 		points[i] = basePointAffine
// 	}

// 	fmt.Printf("g1GenAff %s \n", g1GenAff.String())
// 	fmt.Printf("basePointAffine %s \n", basePointAffine.String())

// 	cpuResultAffine := curve.G1Affine{}
// 	_, err := cpuResultAffine.MultiExp(points, scalars, ecc.MultiExpConfig{NbTasks: 16})
// 	if err != nil {
// 		t.Errorf("cpu msm error: %s", err.Error())
// 	}
// 	fmt.Printf("cpuResultAffine %s \n", cpuResultAffine.String())

// 	d_scalar, err := device.CudaMalloc[fr.Element](0, int(npoints))
// 	if err != nil {
// 		t.Errorf("allocate scalars to device error: %s", err.Error())
// 	}
// 	err = d_scalar.CopyFromHost(scalars)
// 	if err != nil {
// 		t.Errorf("copy scalars to device error: %s", err.Error())
// 	}

// 	d_points, err := device.CudaMalloc[curve.G1Affine](0, int(npoints))
// 	if err != nil {
// 		t.Errorf("allocate points to device error: %s", err.Error())
// 	}
// 	err = d_points.CopyFromHost(points)
// 	if err != nil {
// 		t.Errorf("copy points to device error: %s", err.Error())
// 	}

// 	gpuResultJac := curve.G1Jac{}

// 	cfg := DefaultMSMConfig()
// 	cfg.AreInputsOnDevice = true
// 	cfg.ArePointsInMont = true
// 	cfg.Npoints = npoints
// 	cfg.FfiAffineSz = 64
// 	err = MSM_G1(unsafe.Pointer(&gpuResultJac), d_points.AsPtr(), d_scalar.AsPtr(), 0, cfg)
// 	if err != nil {
// 		t.Errorf("invoke msm gpu error: %s", err.Error())
// 	}
// 	gpuResultAffine := curve.G1Affine{}
// 	gpuResultAffine.FromJacobian(&gpuResultJac)
// 	fmt.Printf("gpuResultAffine: %s \n", gpuResultAffine.String())
// 	assert.True(t, cpuResultAffine.Equal(&gpuResultAffine), "gpu result is incorrect")

// }

// Source: https://github.com/Consensys/gnark-crypto/blob/ef459367f3b5662d69381e74d2874b5fae1729ea/ecc/bn254/multiexp_test.go#L434
// WARNING: this return points that are NOT on the curve and is meant to be use for benchmarking
// purposes only. We don't check that the result is valid but just measure "computational complexity".
//
// Rationale for generating points that are not on the curve is that for large benchmarks, generating
// a vector of different points can take minutes. Using the same point or subset will bias the benchmark result
// since bucket additions in extended jacobian coordinates will hit doubling algorithm instead of add.
func fillBenchBasesG1(samplePoints []curve.G1Affine) {
	var r big.Int
	r.SetString("340444420969191673093399857471996460938405", 10)
	samplePoints[0].ScalarMultiplication(&samplePoints[0], &r)

	one := samplePoints[0].X
	one.SetOne()

	for i := 1; i < len(samplePoints); i++ {
		samplePoints[i].X.Add(&samplePoints[i-1].X, &one)
		samplePoints[i].Y.Sub(&samplePoints[i-1].Y, &one)
	}
}

func fillBenchScalars(sampleScalars []fr.Element) {
	// ensure every words of the scalars are filled
	for i := 0; i < len(sampleScalars); i++ {
		sampleScalars[i].SetRandom()
	}
}