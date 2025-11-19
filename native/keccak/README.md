Based on some comments in the Rust code, Keccak implementation in Plonky2 is based on [tiny-keccak](https://docs.rs/tiny-keccak/latest/tiny_keccak/) crate which is also based on or related to the C-version [keccak-tiny](https://github.com/coruus/keccak-tiny).

Our C version is taken from [saarinen-keccak](https://github.com/coruus/saarinen-keccak/blob/master/readable_keccak/keccak.c).

The CUDA version is a simple adaptation.
