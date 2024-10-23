package msm

import (
	"fmt"
	"math/big"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/okx/cryptography_cuda/wrappers/go/device"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"
)

const deviceId = 0

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
	b.Run(fmt.Sprintf("CPU %d", npoints), func(b *testing.B) {
		b.ResetTimer()
		cpuResult := curve.G1Affine{}
		for i := 0; i < b.N; i++ {
			cpuResult.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{})
		}
	})

	// GPU, include time to copy data to device
	b.Run(fmt.Sprintf("GPU host input %d", npoints), func(b *testing.B) {
		gpuResult := curve.G1Affine{}
		hostScalars := onHost(scalars[:])
		hostPoints := onHost(points[:])
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gnarkMsmG1(&gpuResult, &hostPoints, &hostScalars)
		}
	})

	// GPU, data already on device
	b.Run(fmt.Sprintf("GPU device input %d", npoints), func(b *testing.B) {
		gpuResult := curve.G1Affine{}
		deviceScalars := copyToDevice(scalars[:])
		devicePoints := copyToDevice(points[:])
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gnarkMsmG1(&gpuResult, &devicePoints, &deviceScalars)
		}
		deviceScalars.Free()
		devicePoints.Free()
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
	cpuResult := curve.G1Affine{}
	_, err := cpuResult.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}
	t.Logf("cpuResult %s \n", cpuResult.String())

	// GPU host input
	gpuResult := curve.G1Affine{}
	hostScalars := onHost(scalars[:])
	hostPoints := onHost(points[:])
	gnarkMsmG1(&gpuResult, &hostPoints, &hostScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}

	// GPU device input
	gpuResult = curve.G1Affine{}
	deviceScalars := copyToDevice(scalars[:])
	devicePoints := copyToDevice(points[:])
	gnarkMsmG1(&gpuResult, &devicePoints, &deviceScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}
}

func TestMsmG2(t *testing.T) {
	const npoints uint32 = 1 << 10
	var (
		points  [npoints]curve.G2Affine
		scalars [npoints]fr.Element
	)

	fillRandomScalars(scalars[:])
	fillRandomBasesG2(points[:])

	// CPU
	cpuResult := curve.G2Affine{}
	_, err := cpuResult.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{})
	if err != nil {
		t.Errorf("cpu msm error: %s", err.Error())
	}

	// GPU host input
	gpuResult := curve.G2Affine{}
	hostScalars := onHost(scalars[:])
	hostPoints := onHost(points[:])
	gnarkMsmG2(&gpuResult, &hostPoints, &hostScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}

	// GPU device input
	gpuResult = curve.G2Affine{}
	deviceScalars := copyToDevice(scalars[:])
	devicePoints := copyToDevice(points[:])
	gnarkMsmG2(&gpuResult, &devicePoints, &deviceScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}
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

func fillRandomBasesG2(samplePoints []curve.G2Affine) {
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

func ReverseBytes(arr []byte) {
	for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
		arr[i], arr[j] = arr[j], arr[i]
	}
}

func gnarkMsmG1(res *curve.G1Affine, points *device.HostOrDeviceSlice[curve.G1Affine], scalars *device.HostOrDeviceSlice[fr.Element]) {
	if points.Len() != scalars.Len() {
		panic("points and scalars must have the same length")
	}
	cfg := DefaultMSMConfig()
	cfg.Npoints = uint32(points.Len())
	cfg.ArePointsInMont = true
	cfg.LargeBucketFactor = 2
	if points.IsOnDevice() && scalars.IsOnDevice() {
		cfg.AreInputsOnDevice = true
	} else {
		cfg.AreInputsOnDevice = false
	}
	err := MSM_G1(unsafe.Pointer(res), points.AsPtr(), scalars.AsPtr(), 0, cfg)
	if err != nil {
		panic(err)
	}
}

func gnarkMsmG2(res *curve.G2Affine, points *device.HostOrDeviceSlice[curve.G2Affine], scalars *device.HostOrDeviceSlice[fr.Element]) {
	if points.Len() != scalars.Len() {
		panic("points and scalars must have the same length")
	}
	cfg := DefaultMSMConfig()
	cfg.Npoints = uint32(points.Len())
	cfg.ArePointsInMont = true
	cfg.LargeBucketFactor = 2
	if points.IsOnDevice() && scalars.IsOnDevice() {
		cfg.AreInputsOnDevice = true
	} else {
		cfg.AreInputsOnDevice = false
	}
	err := MSM_G2(unsafe.Pointer(res), points.AsPtr(), scalars.AsPtr(), 0, cfg)
	if err != nil {
		panic(err)
	}
}

func copyToDevice[T any](hostData []T) device.HostOrDeviceSlice[T] {
	deviceSlice, err := device.CudaMalloc[T](deviceId, len(hostData))
	if err != nil {
		panic(err)
	}
	if err := deviceSlice.CopyFromHost(hostData[:]); err != nil {
		panic(err)
	}
	return *deviceSlice
}

func onHost[T any](hostData []T) device.HostOrDeviceSlice[T] {
	deviceSlice := device.NewEmpty[T]()
	deviceSlice.OnHost(hostData)
	return *deviceSlice
}
