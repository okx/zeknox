# Go Wrapper
Download prebuilt files to `/usr/local/lib/`, so that `#cgo LDFLAGS: -L/usr/local/lib` can work
```sh
sudo cp libblst.a libzeknox.a /usr/local/lib/
```

Or, build from source
```sh
# in the repo root directory
cd native/build
cmake .. -DBUILD_MSM=ON -DG2_ENABLED=ON -DCURVE=BN254
cmake --build .
```

## Test
```sh
go test github.com/okx/zeknox/wrappers/go/msm
go test github.com/okx/zeknox/wrappers/go/device

# Benchmark MSM
go test -bench=Msm github.com/okx/zeknox/wrappers/go/msm

# will fail, since currently LDE is not working with BN254
# go test github.com/okx/zeknox/wrappers/go/lib
```
