#!/bin/sh

if ! [ -d "scripts" ]; then
    echo -e "Error: scripts directory not found. Please run this script from the \e[44mnative\e[0m directory of the repository."
    exit 1
fi

CAP=`./scripts/configure.sh | grep capability | cut -d ' ' -f 3`

rm -rf build
if [ -z "$CAP" ]; then
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
else
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DCUDA_ARCH=$CAP
fi
cmake --build build -j