#!/bin/sh

# CAP=`./configure.sh | grep capability | cut -d ' ' -f 3`
CAP="86;89;90"

rm -rf build
if [ -z "$CAP" ]; then
    cmake -B build -DBUILD_TESTS=ON
else
    cmake -B build -DBUILD_TESTS=ON -DCUDA_ARCH=$CAP
fi
cmake --build build -j

if [ "$1" = "-i" ]; then
    sudo cmake --install build
fi
