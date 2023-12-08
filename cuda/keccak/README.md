Based on some comments in the Rust code, Keccak implementation in Plonky2 is based on [tiny-keccak](https://docs.rs/tiny-keccak/latest/tiny_keccak/) crate which is also based on or related to the C-version [keccak-tiny](https://github.com/coruus/keccak-tiny).

However, neither ``sha3_256()`` nor ``shake256()`` from kekkac-tiny produce the same results as the Keccak hasher in Plonky2.

How to test?

```
make test_keccak
./test_kekkac
```

The outputs are different.