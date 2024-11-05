#!/bin/bash

if [ -d $1 ]; then
    exit 0
fi

H=`head -n 1 $1`
if [[ $H =~ "// Copyright" ]]; then
    if [[ $H == "// Copyright 2024 OKX" ]]; then
        echo "Wrong header in file $1"
        exit 1
    fi
else
    echo "Error: no copyright in file $1"
    exit 1
fi

# echo "OK $1"