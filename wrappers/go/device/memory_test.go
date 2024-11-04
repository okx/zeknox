package device

import (
	"testing"
)

const device_alloc_size = 1024

func initArray(n int64) []int {
	arr := make([]int, n)
	for i := range arr {
		arr[i] = i
	}
	return arr
}

func TestHostOrDeviceSlice_Len(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	hostSlice := HostOrDeviceSlice[int]{host: hostArray, isDevice: false}
	deviceSlice := HostOrDeviceSlice[int]{device: struct {
		id int
		s  []int
	}{id: 0, s: hostArray}, isDevice: true}

	if hostSlice.Len() != device_alloc_size {
		t.Errorf("Expected length %d, got %d", device_alloc_size, hostSlice.Len())
	}

	if deviceSlice.Len() != device_alloc_size {
		t.Errorf("Expected length %d, got %d", device_alloc_size, deviceSlice.Len())
	}
}

func TestHostOrDeviceSlice_IsEmpty(t *testing.T) {
	hostSlice := HostOrDeviceSlice[int]{host: []int{}, isDevice: false}
	deviceSlice := HostOrDeviceSlice[int]{device: struct {
		id int
		s  []int
	}{id: 0, s: []int{}}, isDevice: true}

	if !hostSlice.IsEmpty() {
		t.Errorf("Expected host slice to be empty")
	}

	if !deviceSlice.IsEmpty() {
		t.Errorf("Expected device slice to be empty")
	}
}

func TestHostOrDeviceSlice_IsOnDevice(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	hostSlice := HostOrDeviceSlice[int]{host: hostArray, isDevice: false}
	deviceSlice := HostOrDeviceSlice[int]{device: struct {
		id int
		s  []int
	}{id: 0, s: hostArray}, isDevice: true}

	if hostSlice.IsOnDevice() {
		t.Errorf("Expected host slice to be on host")
	}

	if !deviceSlice.IsOnDevice() {
		t.Errorf("Expected device slice to be on device")
	}
}

func TestHostOrDeviceSlice_AsSlice(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	hostSlice := HostOrDeviceSlice[int]{host: hostArray, isDevice: false}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for device slice")
		}
	}()

	deviceSlice := HostOrDeviceSlice[int]{device: struct {
		id int
		s  []int
	}{id: 0, s: hostArray}, isDevice: true}

	_ = deviceSlice.AsSlice()

	if len(hostSlice.AsSlice()) != device_alloc_size {
		t.Errorf("Expected length %d, got %d", device_alloc_size, len(hostSlice.AsSlice()))
	}
}

func TestHostOrDeviceSlice_AsPtr(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	hostSlice := HostOrDeviceSlice[int]{host: hostArray, isDevice: false}
	deviceSlice := HostOrDeviceSlice[int]{device: struct {
		id int
		s  []int
	}{id: 0, s: hostArray}, isDevice: true}

	if hostSlice.AsPtr() == nil {
		t.Errorf("Expected non-nil pointer for host slice")
	}

	if deviceSlice.AsPtr() == nil {
		t.Errorf("Expected non-nil pointer for device slice")
	}
}

func TestHostOrDeviceSlice_OnHost(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	hostSlice := HostOrDeviceSlice[int]{}
	hostSlice.OnHost(hostArray)

	if len(hostSlice.host) != device_alloc_size {
		t.Errorf("Expected length %d, got %d", device_alloc_size, len(hostSlice.host))
	}

	if hostSlice.isDevice {
		t.Errorf("Expected isDevice to be false")
	}
}

func TestCudaMalloc(t *testing.T) {
	slice, err := CudaMalloc[int](0, device_alloc_size)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if slice == nil {
		t.Errorf("Expected non-nil slice")
	}

	if !slice.isDevice {
		t.Errorf("Expected isDevice to be true")
	}

	if len(slice.device.s) != device_alloc_size {
		t.Errorf("Expected length %d, got %d", device_alloc_size, len(slice.device.s))
	}

	slice.Free()
}

func TestHostOrDeviceSlice_CopyFromHost(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	slice, err := CudaMalloc[int](0, device_alloc_size)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	defer slice.Free()

	err = slice.CopyFromHost(hostArray)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
}

func TestHostOrDeviceSlice_CopyToHost(t *testing.T) {
	hostArray := initArray(device_alloc_size)
	slice, err := CudaMalloc[int](0, device_alloc_size)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	defer slice.Free()

	err = slice.CopyFromHost(hostArray)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	hostSlice := make([]int, device_alloc_size)
	err = slice.CopyToHost(hostSlice)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	for i, v := range hostSlice {
		if v != i {
			t.Errorf("Expected %d, got %d", i+1, v)
		}
	}
}
