#ifndef GLOBAL_LOAD_VQF_H
#define GLOBAL_LOAD_VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/warp_storage_block.cuh"

#include "include/hash_metadata.cuh"
 


//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) optimized_vqf {


	storage_block * blocks;

	uint64_t num_blocks;
	
	uint * counters;



	int seed;

	__device__ bool insert_item(uint64_t item, int warpID);

	__device__ bool query_item(uint64_t item, int warpID);


	__host__ void bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses);


	__host__ void bulk_query(uint64_t * items, uint64_t nitems, bool * hits);


	__device__ uint64_t get_hash(uint64_t item);

} optimized_vqf;


__host__ optimized_vqf * build_vqf(uint64_t nslots);

__host__ void free_vqf(optimized_vqf * vqf);

#endif