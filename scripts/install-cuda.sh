#!/bin/bash

sudo apt update
sudo apt -y install lsb-base lsb-release

ARCH=$(uname -m)
OS=$(lsb_release -rs | tr '[:upper:]' '[:lower:]')

if [[ "$OS" != "24.04" && "$ARCH" != "x86_64" ]]; then
    echo "Unsupported OS version: ${OS} or architecture ${ARCH}."
    exit 1
fi

echo "Installing CUDA toolkit for ${OS} on ${ARCH} architecture..."

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

rm -f cuda-keyring_1.1-1_all.deb