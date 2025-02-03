#!/usr/bin/env bash

./build_gmp.sh host
cd ../../native/depends/blst && sh build.sh
