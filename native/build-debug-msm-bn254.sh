#!/bin/sh

CAP=`./configure.sh | grep capability | cut -d ' ' -f 3`

rm -rf build
if [ -z "$CAP" ]; then
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCURVE=BN254 -DBUILD_MSM=ON -DG2_ENABLED=ON
else
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCH=$CAP -DCURVE=BN254 -DBUILD_MSM=ON -DG2_ENABLED=ON
fi
cmake --build build -j