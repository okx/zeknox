#!/bin/sh

mkdir -p build && cd build
rm -rf ./*
cmake -DCMAKE_BUILD_TYPE=Debug .. -DBUILD_TESTS=ON
make -j4