# Build

```
make lib
```

## Flags

``USE_CUDA`` - enables the cuda code (on by default)

``FEATURE_GOLDILOCKS`` - enables Goldilocks field support in NTT (on by default)

``EXPOSE_C_INTERFACE`` - expose C interface to Rust (on by default)

``TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE`` - error messages are handled in the native code

``__USE_AVX__`` - enable AVX (AVX2) code [currently only for Poseidon hashing]

``__AVX512__`` - enable AVX512 code and requires ``__USE_AVX__`` [currently only for Poseidon hashing]

# Tests

```
make tests.exe
./tests.exe
```

