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

func TestMsmG2InputsNotOnDevice(t *testing.T) {
	var npoints uint32 = 1024
	point_scale_factor := 1
	scalars := make([]fr.Element, npoints)
	for i := 0; i < len(scalars); i++ {
		scalars[i] = fr.NewElement(uint64(i + 3))
	}

	_, _, _, g2GenAff := curve.Generators()
	basePointAffine := curve.G2Affine{}
	basePointAffine.ScalarMultiplication(&g2GenAff, big.NewInt(int64(point_scale_factor)))

	points := make([]curve.G2Affine, npoints)
	for i := 0; i < len(points); i++ {
		points[i] = basePointAffine
	}

	cpuResultAffine := curve.G2Affine{}
	_, err := cpuResultAffine.MultiExp(points, scalars, ecc.MultiExpConfig{NbTasks: 16})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}

	// fmt.Printf("cpuResultAffine string %v \n", cpuResultAffine.String())
	// fmt.Printf("cpuResultAffine x.a0 bytes %x \n", cpuResultAffine.X.A0.Bytes())
	// fmt.Printf("cpuResultAffine x.a1 bytes %x \n", cpuResultAffine.X.A1.Bytes())
	// fmt.Printf("cpuResultAffine y.a0 bytes %x \n", cpuResultAffine.Y.A0.Bytes())
	// fmt.Printf("cpuResultAffine y.a0 bytes %x \n", cpuResultAffine.Y.A1.Bytes())

	data := Alloc_c(128)
	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = false
	cfg.Npoints = npoints
	cfg.ArePointsInMont = true
	cfg.LargeBucketFactor = 2
	err = MSM_G2(data, unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), 0, cfg)
	if err != nil {
		t.Errorf("invoke msm gpu error: %s", err.Error())
	}

	byteArray := (*[128]byte)(data)
	fmt.Printf("%x\n", byteArray)

	gpuG2Aff := curve.G2Affine{}
	ReverseBytes(byteArray[0:32])
	gpuG2Aff.X.A0.SetBytes(byteArray[0:32])

	ReverseBytes(byteArray[32:64])
	gpuG2Aff.X.A1.SetBytes(byteArray[32:64])

	ReverseBytes(byteArray[64:96])
	gpuG2Aff.Y.A0.SetBytes(byteArray[64:96])

	ReverseBytes(byteArray[96:128])
	gpuG2Aff.Y.A1.SetBytes(byteArray[96:128])

	assert.True(t, cpuResultAffine.Equal(&gpuG2Aff), "gpu result is incorrect")

}

func TestMsmG2InputsOnDevice(t *testing.T) {

	var npoints uint32 = 4096
	point_scale_factor := 10
	scalars := make([]fr.Element, npoints)
	for i := 0; i < len(scalars); i++ {
		scalars[i] = fr.NewElement(uint64(i + 1))
	}

	_, _, _, g2GenAff := curve.Generators()
	basePointAffine := curve.G2Affine{}
	basePointAffine.ScalarMultiplication(&g2GenAff, big.NewInt(int64(point_scale_factor)))

	points := make([]curve.G2Affine, npoints)
	for i := 0; i < len(points); i++ {
		points[i] = basePointAffine
	}

	cpuResultAffine := curve.G2Affine{}
	_, err := cpuResultAffine.MultiExp(points, scalars, ecc.MultiExpConfig{NbTasks: 16})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}

	d_points, err := device.CudaMalloc[curve.G2Affine](0, int(npoints))
	if err != nil {
		t.Errorf("allocate points to device error: %s", err.Error())
	}
	err = d_points.CopyFromHost(points)
	if err != nil {
		t.Errorf("copy points to device error: %s", err.Error())
	}

	d_scalars, err := device.CudaMalloc[fr.Element](0, int(npoints))
	if err != nil {
		t.Errorf("allocate points to device error: %s", err.Error())
	}
	err = d_scalars.CopyFromHost(scalars)
	if err != nil {
		t.Errorf("copy scalars to device error: %s", err.Error())
	}

	data := Alloc_c(128)
	cfg := DefaultMSMConfig()
	cfg.AreInputsOnDevice = true
	cfg.Npoints = npoints
	cfg.ArePointsInMont = true
	cfg.LargeBucketFactor = 2
	err = MSM_G2(data, d_points.AsPtr(), d_scalars.AsPtr(), 0, cfg)
	if err != nil {
		t.Errorf("invoke msm gpu error: %s", err.Error())
	}

	byteArray := (*[128]byte)(data)
	fmt.Printf("%x\n", byteArray)

	gpuG2Aff := curve.G2Affine{}
	ReverseBytes(byteArray[0:32])
	gpuG2Aff.X.A0.SetBytes(byteArray[0:32])

	ReverseBytes(byteArray[32:64])
	gpuG2Aff.X.A1.SetBytes(byteArray[32:64])

	ReverseBytes(byteArray[64:96])
	gpuG2Aff.Y.A0.SetBytes(byteArray[64:96])

	ReverseBytes(byteArray[96:128])
	gpuG2Aff.Y.A1.SetBytes(byteArray[96:128])

	assert.True(t, cpuResultAffine.Equal(&gpuG2Aff), "gpu result is incorrect")

}

func ReverseBytes(arr []byte) {
	for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
		arr[i], arr[j] = arr[j], arr[i]
	}
}
