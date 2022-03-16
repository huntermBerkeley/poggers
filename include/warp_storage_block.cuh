#ifndef WARP_STORAGE_H
#define WARP_STORAGE_H



#include <cuda.h>
#include <cuda_runtime_api.h>

#include "include/hash_metadata.cuh"

#define SLOTS_PER_WARP_STORAGE 32
#define KEY_EMPTY 0

//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) storage_block {





	#if TAG_BITS == 8

	uint8_t slots[32];

	__device__ bool insert(uint8_t item, int warpID);

	__device__ bool query (uint8_t item, int warpID);

	#elif TAG_BITS == 16

	uint16_t slots[32];

	__device__ bool insert(uint16_t item, int warpID);
 
	__device__ bool query (uint16_t item, int warpID);

	#elif TAG_BITS == 32

	uint slots[32];

	__device__ bool insert(uint item, int warpID);

	__device__ bool query (uint item, int warpID);

	#elif TAG_BITS == 64

	uint64_t slots[32];

	__device__ bool insert(uint64_t item, int warpID);

	__device__ bool query (uint64_t item, int warpID);

	#endif





} storage_block;



#endif