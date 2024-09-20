#!/usr/bin/env bash

./build_gmp.sh host
cd depends/blst && sh build.sh && cd ..