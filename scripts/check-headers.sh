#!/bin/bash

# native
find ../native \( -path ../native/depends -o -path ../native/build -o -path ../native/utils/deviceQuery \) -prune -o -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | xargs -n1 ./check-header.sh

# wrappers/cpp
find ../wrappers/cpp \( -path ../wrappers/cpp/depends -o -path ../wrappers/cpp/build -o -path ../wrappers/cpp/field \) -prune -o -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | xargs -n1 ./check-header.sh

# wrappers/rust
find ../wrappers/rust \( -path ../wrappers/rust/target \) -prune -o -name "*.rs" -a -not -name "bindings.rs" | xargs -n1 ./check-header.sh

# wrappers/go
find ../wrappers/go -name "*.go" | xargs -n1 ./check-header.sh