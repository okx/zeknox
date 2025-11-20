#!/bin/sh

if ! [ -d "scripts" ]; then
    echo -e "Error: scripts directory not found. Please run this script from the \e[44mnative\e[0m directory of the repository."
    exit 1
fi

# CAP=`./scripts/configure.sh | grep capability | cut -d ' ' -f 3`
CAP="86;89;90"

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
