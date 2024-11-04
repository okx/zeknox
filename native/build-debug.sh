#!/bin/sh

CAP=`./configure.sh | grep capability | cut -d ' ' -f 3`

mkdir -p build && cd build
rm -rf ./*
if [ -z "$CAP" ]; then
cmake -DCMAKE_BUILD_TYPE=Debug .. -DBUILD_TESTS=ON
else
cmake -DCMAKE_BUILD_TYPE=Debug .. -DBUILD_TESTS=ON -DCUDA_ARCH=$CAP
fi
make -j