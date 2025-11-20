#!/bin/bash

if ! [ -d "scripts" ]; then
    echo -e "Error: scripts directory not found. Please run this script from the \e[44mnative\e[0m directory of the repository."
    exit 1
fi

cd utils/deviceQuery
make
if ! [ -e deviceQuery ]; then
    echo "Error buidling CUDA deviceQuery!"
    cd ../..
    exit 1
fi

CAP=`./deviceQuery | grep "CUDA Capability" | head -n 1 | tr -d ' ' | cut -d ':' -f 2 | tr -d '.'`
if [ -z "$CAP" ]; then
    echo "Unable to get CUDA capability on this system!"
    cd ../..
    exit 1
fi

cd ../..

echo "CUDA capability: $CAP"
echo "Usage: cmake .. -DCUDA_ARCH=$CAP"