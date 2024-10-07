# Native (C/C++) usage
## Build
```
git submodule init
git submodule update
sh build_dep.sh
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./package -DCURVE=GOLDILOCKS -DG2_ENABLED=ON
# or
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=./package -DCURVE=BN254 -DG2_ENABLED=OFF
make -j4 VERBOSE=1 & make install
```

to enable G2
```
-DG2_ENABLED=ON
```

## Test
- run test
```
export LD_LIBRARY_PATH=package/lib/ # if building a shared lib
./test_bn254 --gtest_filter=*msm_bn254_g1_curve_gpu_consistency_with_cpu*
./test_gl64 --gtest_filter=*xxx*
```

## Remarks
- current cuda build time for G2_ENABLED is very slow. therefore G2_ENABLED is not enabled by default.