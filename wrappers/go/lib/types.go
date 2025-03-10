// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

package lib

type NTTInputOutputOrder int

const (
	NN NTTInputOutputOrder = iota
	NR
	RN
	RR
)

type NTTDirection int

const (
	Forward NTTDirection = iota
	Inverse
)

type NTTType int

const (
	Standard NTTType = iota
	Coset
)

type NTTConfig struct {
	Batches            uint32
	Order              NTTInputOutputOrder
	NttType            NTTType
	ExtensionRateBits  uint32
	AreInputsOnDevice  bool
	AreOutputsOnDevice bool
	WithCoset          bool
	IsMultiGPU         bool
	SaltSize           uint32
}

func DefaultNTTConfig() NTTConfig {
	return NTTConfig{
		Batches:            1,
		Order:              NN,
		NttType:            Standard,
		ExtensionRateBits:  0,
		AreInputsOnDevice:  false,
		AreOutputsOnDevice: false,
		WithCoset:          false,
		IsMultiGPU:         false,
		SaltSize:           0,
	}
}

type TransposeConfig struct {
	Batches            uint32
	AreInputsOnDevice  bool
	AreOutputsOnDevice bool
}

func DefaultTransposeConfig() TransposeConfig {
	return TransposeConfig{
		Batches:            1,
		AreInputsOnDevice:  false,
		AreOutputsOnDevice: false,
	}
}

const (
	HashPoseidon int = 0
	HashKeccak
	HashPoseidonBN128
	HashPoseidon2
	HashMonolith
)
