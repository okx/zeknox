#!/bin/bash

VER="v1.0.1"

if [ $# -lt 1 ]; then
    echo "Usage: $0 gl64 | bn254"
    echo " - use gl64 to install zeknox with Goldilocks"
    echo " - use bn254 to install zeknox with BN254"
    exit 1
fi

if [ "$1" == "gl64" ]; then
    echo "Installing zeknox ${VER} with Goldilocks"
    wget https://github.com/okx/zeknox/releases/download/${VER}/gl64-86-89-90-libzeknox.a
    wget https://github.com/okx/zeknox/releases/download/${VER}/libblst.a
    sudo cp gl64-86-89-90-libzeknox.a /usr/local/lib/libzeknox.a
    sudo cp libblst.a /usr/local/lib/
elif [ "$1" == "bn254" ]; then
    echo "Installing zeknox ${VER} with BN254"
    wget https://github.com/okx/zeknox/releases/download/${VER}/bn254-msm-86-89-90-libzeknox.a
    wget https://github.com/okx/zeknox/releases/download/${VER}/libblst.a
    sudo cp bn254-msm-86-89-90-libzeknox.a /usr/local/lib/libzeknox.a
    sudo cp libblst-bn254.a /usr/local/lib/
else
    echo "Invalid argument. Use 'gl64' or 'bn254'."
    exit 1
fi
