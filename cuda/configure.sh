#!/bin/bash

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

echo "CUDA_ARCH = sm_$CAP" | tee CudaArch.mk