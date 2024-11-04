#!/bin/sh

mkdir -p build && cd build
rm -rf ./*
cmake .. -DBUILD_TESTS=ON
make -j4