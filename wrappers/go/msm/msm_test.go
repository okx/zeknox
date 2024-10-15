package msm

import (
	"fmt"
	"math/big"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"
	"zeknox/device"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"
)

func BenchmarkMsmG1(b *testing.B) {
	// Gnark 300w constraint MSM G1 total size
	const npoints uint32 = 1<<23 + 1<<22
	var (
		points  [npoints]curve.G1Affine
		scalars [npoints]fr.Element
	)
	fillRandomScalars(scalars[:])
	fillBenchBasesG1(points[:])

	// CPU
	cpuNum := runtime.NumCPU()
	b.Run(fmt.Sprintf("CPU %d", npoints), func(b *testing.B) {
		b.ResetTimer()
		cpuResultAffine := curve.G1Affine{}
		for i := 0; i < b.N; i++ {
			// NbTasks same as Gnark prover
			cpuResultAffine.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{NbTasks: cpuNum / 2})
		}
	})

	// GPU config
	cfg := DefaultMSMConfig()
	cfg.ArePointsInMont = true
	cfg.Npoints = npoints
	cfg.FfiAffineSz = 64

	// GPU, include time to copy data to device
	b.Run(fmt.Sprintf("GPU host input %d", npoints), func(b *testing.B) {
		cfg.AreInputsOnDevice = false
		gpuResultJac := curve.G1Jac{}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MSM_G1(unsafe.Pointer(&gpuResultJac), unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), 0, cfg)
		}
	})

	// GPU, data already on device
	b.Run(fmt.Sprintf("GPU device input %d", npoints), func(b *testing.B) {
		cfg.AreInputsOnDevice = true
		// copy scalars
		d_scalar, err := device.CudaMalloc[fr.Element](0, int(npoints))
		if err != nil {
			b.Errorf("allocate scalars to device error: %s", err.Error())
		}
		err = d_scalar.CopyFromHost(scalars[:])
		if err != nil {
			b.Errorf("copy scalars to device error: %s", err.Error())
		}
		// copy points
		d_points, err := device.CudaMalloc[curve.G1Affine](0, int(npoints))
		if err != nil {
			b.Errorf("allocate points to device error: %s", err.Error())
		}
		err = d_points.CopyFromHost(points[:])
		if err != nil {
			b.Errorf("copy points to device error: %s", err.Error())
		}
		gpuResultJac := curve.G1Jac{}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			MSM_G1(unsafe.Pointer(&gpuResultJac), d_points.AsPtr(), d_scalar.AsPtr(), 0, cfg)
		}
	})
}

func TestMsmG1(t *testing.T) {
	const npoints uint32 = 1 << 10
	var (
		points  [npoints]curve.G1Affine
		scalars [npoints]fr.Element
	)
	fillRandomScalars(scalars[:])
	fillRandomBasesG1(points[:])

	// CPU
	cpuResultAffine := curve.G1Affine{}
	_, err := cpuResultAffine.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{NbTasks: 16})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}
	t.Logf("cpuResultAffine %s \n", cpuResultAffine.String())

	// GPU host input
	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = false
	cfg.ArePointsInMont = true
	cfg.Npoints = npoints
	cfg.FfiAffineSz = 64
	gpuResultJac := curve.G1Jac{}
	MSM_G1(unsafe.Pointer(&gpuResultJac), unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), 0, cfg)
	gpuResultAffine := curve.G1Affine{}
	gpuResultAffine.FromJacobian(&gpuResultJac)
	t.Logf("gpuResultAffine: %s \n", gpuResultAffine.String())
	assert.True(t, cpuResultAffine.Equal(&gpuResultAffine), "gpu result is incorrect")

	// GPU device input
	d_scalar, err := device.CudaMalloc[fr.Element](0, int(npoints))
	if err != nil {
		t.Errorf("allocate scalars to device error: %s", err.Error())
	}
	err = d_scalar.CopyFromHost(scalars[:])
	if err != nil {
		t.Errorf("copy scalars to device error: %s", err.Error())
	}
	d_points, err := device.CudaMalloc[curve.G1Affine](0, int(npoints))
	if err != nil {
		t.Errorf("allocate points to device error: %s", err.Error())
	}
	err = d_points.CopyFromHost(points[:])
	if err != nil {
		t.Errorf("copy points to device error: %s", err.Error())
	}
	cfg.AreInputsOnDevice = true
	err = MSM_G1(unsafe.Pointer(&gpuResultJac), d_points.AsPtr(), d_scalar.AsPtr(), 0, cfg)
	if err != nil {
		t.Errorf("invoke msm gpu error: %s", err.Error())
	}
	gpuResultAffine.FromJacobian(&gpuResultJac)
	t.Logf("gpuResultAffine: %s \n", gpuResultAffine.String())
	assert.True(t, cpuResultAffine.Equal(&gpuResultAffine), "gpu result is incorrect")
}

func fillRandomScalars(sampleScalars []fr.Element) {
	// ensure every words of the scalars are filled
	for i := 0; i < len(sampleScalars); i++ {
		sampleScalars[i].SetRandom()
	}
}

func fillRandomBasesG1(samplePoints []curve.G1Affine) {
	for i := 0; i < len(samplePoints); i++ {
		randomInt := big.NewInt(rand.Int63())
		samplePoints[i].ScalarMultiplicationBase(randomInt)
	}
}

// WARNING: this return points that are NOT on the curve and is meant to be use for benchmarking
// purposes only. We don't check that the result is valid but just measure "computational complexity".
//
// Rationale for generating points that are not on the curve is that for large benchmarks, generating
// a vector of different points can take minutes. Using the same point or subset will bias the benchmark result
// since bucket additions in extended jacobian coordinates will hit doubling algorithm instead of add.
//
// Source: https://github.com/Consensys/gnark-crypto/blob/ef459367f3b5662d69381e74d2874b5fae1729ea/ecc/bn254/multiexp_test.go#L434
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
