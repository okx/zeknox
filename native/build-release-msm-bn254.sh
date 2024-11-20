#!/bin/sh

CAP=`./configure.sh | grep capability | cut -d ' ' -f 3`

rm -rf build
if [ -z "$CAP" ]; then
    cmake -B build -DCURVE=BN254 -DBUILD_MSM=ON -DG2_ENABLED=ON
else
    cmake -B build -DCUDA_ARCH=$CAP -DCURVE=BN254 -DBUILD_MSM=ON -DG2_ENABLED=ON
fi
cmake --build build -j

if [ "$1" = "-i" ]; then
    sudo cmake --install build
fi