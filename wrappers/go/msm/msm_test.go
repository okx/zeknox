package msm

import (
	"context"
	"fmt"
	"math/big"
	"math/rand"
	"testing"
	"time"
	"unsafe"

	"github.com/okx/cryptography_cuda/wrappers/go/device"
	"golang.org/x/sync/errgroup"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"
)

const deviceId = 1

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
			gnarkMsm(&gpuResult, &hostPoints, &hostScalars)
		}
	})

	// GPU, data already on device
	b.Run(fmt.Sprintf("GPU device input %d", npoints), func(b *testing.B) {
		gpuResult := curve.G1Affine{}
		deviceScalars := copyToDevice(scalars[:])
		devicePoints := copyToDevice(points[:])
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gnarkMsm(&gpuResult, &devicePoints, &deviceScalars)
		}
		deviceScalars.Free()
		devicePoints.Free()
	})
}

func BenchmarkMsmG2(b *testing.B) {
	const npoints uint32 = 1<<22
	var (
		points  [npoints]curve.G2Affine
		scalars [npoints]fr.Element
	)
	fillRandomScalars(scalars[:])
	fillBenchBasesG2(points[:])

	// CPU
	b.Run(fmt.Sprintf("CPU %d", npoints), func(b *testing.B) {
		b.ResetTimer()
		cpuResult := curve.G2Affine{}
		for i := 0; i < b.N; i++ {
			cpuResult.MultiExp(points[:], scalars[:], ecc.MultiExpConfig{})
		}
	})

	// GPU, include time to copy data to device
	b.Run(fmt.Sprintf("GPU host input %d", npoints), func(b *testing.B) {
		gpuResult := curve.G2Affine{}
		hostScalars := onHost(scalars[:])
		hostPoints := onHost(points[:])
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gnarkMsm(&gpuResult, &hostPoints, &hostScalars)
		}
	})

	// GPU, data already on device
	b.Run(fmt.Sprintf("GPU device input %d", npoints), func(b *testing.B) {
		gpuResult := curve.G2Affine{}
		deviceScalars := copyToDevice(scalars[:])
		devicePoints := copyToDevice(points[:])
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gnarkMsm(&gpuResult, &devicePoints, &deviceScalars)
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
	gnarkMsm(&gpuResult, &hostPoints, &hostScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}

	// GPU device input
	gpuResult = curve.G1Affine{}
	deviceScalars := copyToDevice(scalars[:])
	devicePoints := copyToDevice(points[:])
	gnarkMsm(&gpuResult, &devicePoints, &deviceScalars)
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
	gnarkMsm(&gpuResult, &hostPoints, &hostScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}

	// GPU device input
	gpuResult = curve.G2Affine{}
	deviceScalars := copyToDevice(scalars[:])
	devicePoints := copyToDevice(points[:])
	gnarkMsm(&gpuResult, &devicePoints, &deviceScalars)
	if !assert.True(t, cpuResult.Equal(&gpuResult), "gpu result is incorrect") {
		t.Logf("cpuResult: %s", cpuResult.String())
		t.Logf("gpuResult: %s", gpuResult.String())
	}
}

func TestMsmG1G2ReusePointer(t *testing.T) {
	const npoints uint32 = 1 << 10
	var (
		pointsG1  [npoints]curve.G1Affine
		pointsG2  [npoints]curve.G2Affine
		scalarsG1 [npoints]fr.Element
		scalarsG2 [npoints]fr.Element
	)
	fillRandomScalars(scalarsG1[:])
	fillRandomScalars(scalarsG2[:])
	fillRandomBasesG1(pointsG1[:])
	fillRandomBasesG2(pointsG2[:])

	g, _ := errgroup.WithContext(context.Background())
	testG1 := func(cfg MSMConfig, cpuG1Result curve.G1Affine, deviceScalars *device.HostOrDeviceSlice[fr.Element], devicePoints *device.HostOrDeviceSlice[curve.G1Affine]) error {
		gpuResult := curve.G1Affine{}
		gnarkMsm(&gpuResult, devicePoints, deviceScalars, cfg)
		if !assert.True(t, cpuG1Result.Equal(&gpuResult), "gpu result is incorrect") {
			return fmt.Errorf("gpu result is incorrect")
		}
		return nil
	}
	testG2 := func(cfg MSMConfig, cpuG2Result curve.G2Affine, deviceScalars *device.HostOrDeviceSlice[fr.Element], devicePoints *device.HostOrDeviceSlice[curve.G2Affine]) error {
		gpuResult := curve.G2Affine{}
		gnarkMsm(&gpuResult, devicePoints, deviceScalars, cfg)
		if !assert.True(t, cpuG2Result.Equal(&gpuResult), "gpu result is incorrect") {
			return fmt.Errorf("gpu result is incorrect")
		}
		return nil
	}

	// Reuse the same points for multiple msm calls
	// simulate gnark prover in multiple rounds
	deviceGlobalG1 := copyToDevice(pointsG1[:])
	deviceGlobalG2 := copyToDevice(pointsG2[:])
	cfg := DefaultMSMConfig()
	cfg.AreInputPointInMont = true
	cfg.AreInputScalarInMont = true
	for i := 0; i < 3; i++ {
		if i == 0 {
			// points in 1st round are in montgomery form
			cfg.AreInputPointInMont = true
		} else {
			// After 1st round, points are converted to affine form
			cfg.AreInputPointInMont = false
		}
		cpuG1Result := curve.G1Affine{}
		cpuG1Result.MultiExp(pointsG1[:], scalarsG1[:], ecc.MultiExpConfig{})
		cpuG2Result := curve.G2Affine{}
		cpuG2Result.MultiExp(pointsG2[:], scalarsG2[:], ecc.MultiExpConfig{})
		deviceScalarsG1 := copyToDevice(scalarsG1[:])
		deviceScalarsG2 := copyToDevice(scalarsG2[:])
		g.Go(func() error {
			return testG1(cfg, cpuG1Result, &deviceScalarsG1, &deviceGlobalG1)
		})
		g.Go(func() error {
			return testG2(cfg, cpuG2Result, &deviceScalarsG2, &deviceGlobalG2)
		})
		if err := g.Wait(); err != nil {
			t.Errorf("error: %s", err.Error())
		}
	}
}

func fillRandomScalars(sampleScalars []fr.Element) {
	// ensure every words of the scalars are filled
	for i := 0; i < len(sampleScalars); i++ {
		sampleScalars[i].SetRandom()
	}
}

func fillRandomBasesG1(samplePoints []curve.G1Affine) {
	maxScalar := fr.Modulus()
	randSource := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < len(samplePoints); i++ {
		randomInt := new(big.Int).Rand(randSource, maxScalar)
		samplePoints[i].ScalarMultiplicationBase(randomInt)
	}
}

func fillRandomBasesG2(samplePoints []curve.G2Affine) {
	maxScalar := fr.Modulus()
	randSource := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < len(samplePoints); i++ {
		randomInt := new(big.Int).Rand(randSource, maxScalar)
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

func fillBenchBasesG2(samplePoints []curve.G2Affine) {
	var r big.Int
	r.SetString("340444420969191673093399857471996460938405", 10)
	samplePoints[0].ScalarMultiplication(&samplePoints[0], &r)

	one := samplePoints[0].X.A0
	one.SetOne()

	for i := 1; i < len(samplePoints); i++ {
		samplePoints[i].X.A0.Add(&samplePoints[i-1].X.A0, &one)
		samplePoints[i].X.A1.Add(&samplePoints[i-1].X.A1, &one)
		samplePoints[i].Y.A0.Sub(&samplePoints[i-1].Y.A0, &one)
		samplePoints[i].Y.A1.Sub(&samplePoints[i-1].Y.A1, &one)
	}
}

func ReverseBytes(arr []byte) {
	for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
		arr[i], arr[j] = arr[j], arr[i]
	}
}

func gnarkMsm[T curve.G1Affine | curve.G2Affine](res *T, points *device.HostOrDeviceSlice[T], scalars *device.HostOrDeviceSlice[fr.Element], optionalCfg ...MSMConfig) {
	if points.Len() != scalars.Len() {
		panic("points and scalars must have the same length")
	}
	var cfg MSMConfig
	if len(optionalCfg) > 0 {
		// alter AreInputPointInMont and AreInputScalarInMont
		cfg = optionalCfg[0]
	} else {
		cfg = DefaultMSMConfig()
		cfg.AreInputPointInMont = true
		cfg.AreInputScalarInMont = true
	}
	cfg.Npoints = uint32(points.Len())
	cfg.AreOutputPointInMont = true
	cfg.LargeBucketFactor = 2
	if points.IsOnDevice() && scalars.IsOnDevice() {
		cfg.AreInputsOnDevice = true
	} else {
		cfg.AreInputsOnDevice = false
	}
	// Use type assertion to call the appropriate function
    switch any(res).(type) {
    case *curve.G1Affine:
        err := MSM_G1(unsafe.Pointer(res), points.AsPtr(), scalars.AsPtr(), deviceId, cfg)
        if err != nil {
            panic(err)
        }
    case *curve.G2Affine:
        err := MSM_G2(unsafe.Pointer(res), points.AsPtr(), scalars.AsPtr(), deviceId, cfg)
        if err != nil {
            panic(err)
        }
	default:
		panic("unsupported type")
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
