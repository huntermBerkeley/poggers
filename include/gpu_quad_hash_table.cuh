#ifndef GPU_QUAD_HASH_TABLE_H
#define GPU_QUAD_HASH_TABLE_H



#include <cuda.h>
#include <cuda_runtime_api.h>

#include "include/hash_metadata.cuh"


//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) quad_hash_table {




	uint64_t num_slots;

	#if TAG_BITS == 8

	uint8_t * slots;

	__device__ bool insert(uint8_t item);

	__device__ bool query (uint8_t item);

	#elif TAG_BITS == 16

	uint16_t * slots;

	__device__ bool insert(uint16_t item);

	__device__ bool query (uint16_t item);

	#elif TAG_BITS == 32

	uint * slots;

	__device__ bool insert(uint item);

	__device__ bool query (uint item);

	#elif TAG_BITS == 64

	uint64_t * slots;

	__device__ bool insert(uint64_t item);

	__device__ bool query (uint64_t item);

	#endif

	int seed;






} quad_hash_table;


__host__ quad_hash_table * build_hash_table(uint64_t max_nitems);


#endif