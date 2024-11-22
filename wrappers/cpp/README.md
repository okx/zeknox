# Native (C/C++) usage

## Build
```
$ git submodule init
$ git submodule update
$ cd native
$ CAP=`./configure.sh | grep "CUDA capability" | cut -d ' ' -f 3`
$ cd wrappers/cpp
$ sh build_dep.sh
$ mkdir -p build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./package -DCUDA_ARCH=$CAP -DCURVE=BN254 -DBUILD_MSM=ON -DG2_ENABLED=ON
$ make -j
$ make install
```

to enable G2
```
-DG2_ENABLED=ON
```

## Test
- run test
```
./test_bn254 --gtest_filter=*msm_inputs_not_on_device_bn254_g1_curve_gpu_consistency_with_cpu*
./test_bn254 --gtest_filter=*msm_inputs_on_device_bn254_g1_curve_gpu_consistency_with_cpu*
./test_bn254
```

## Remarks
- Current cuda build when G2_ENABLED=ON is very slow (a few minutes). Therefore, G2_ENABLED is not enabled by default.