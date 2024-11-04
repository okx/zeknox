#!/bin/sh

CAP=`./configure.sh | grep capability | cut -d ' ' -f 3`

mkdir -p build && cd build
rm -rf ./*
if [ -z "$CAP" ]; then
cmake .. -DBUILD_TESTS=ON
else
cmake .. -DBUILD_TESTS=ON -DCUDA_ARCH=$CAP
fi
make -j