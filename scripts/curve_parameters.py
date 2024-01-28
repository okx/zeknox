# import blst

## for curve bls_12_381
# Generatore for the field
# PRIMITIVE_ROOT = 5
# MODULUS = 52435875175126190479447740508185965837690552500527637822603658699938581184513 # curve_order

## Goldilocks Field
PRIMITIVE_ROOT = 7
MODULUS = 18446744069414584321 # curve_order, # 2^64 - 2^32 + 1

# P1_INF = blst.P1(bytes.fromhex("400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))
# P2_INF = blst.P2(bytes.fromhex("400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))

## BN254
# MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583 
# PRIMITIVE_ROOT = 0