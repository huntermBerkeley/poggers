#ifndef VQF_H
#define VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/vqf_team_block.cuh"


//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) vqf {


	uint64_t num_blocks;
	
	uint64_t ** buffers;

	uint64_t * buffer_sizes;

	int * locks;

	vqf_block * blocks;

	int seed;

	__device__ void lock_block(int warpID, uint64_t lock);

	__device__ void unlock_block(int warpID, uint64_t lock);


	__device__ void lock_blocks(int warpID, uint64_t lock1, uint64_t lock2);

	__device__ void unlock_blocks(int warpId, uint64_t lock1, uint64_t lock2);


	__device__ bool insert(int warpID, uint64_t item);

	__device__ bool query(int warpID, uint64_t item);

	__device__ bool remove(int warpID, uint64_t item);

	__host__ void attach_buffers(uint64_t * vals, uint64_t nvals);

	//__global__ void set_buffers_binary(uint64_t num_keys, uint64_t * keys);

	//__global__ void set_buffer_lens(uint64_t num_keys, uint64_t * keys);


	__device__ uint64_t hash_key(uint64_t key);

	__device__ bool buffer_insert(int warpID, uint64_t buffer);

	__host__ uint64_t get_num_buffers();


} vqf;




__host__ vqf * build_vqf(uint64_t nitems);


#endif