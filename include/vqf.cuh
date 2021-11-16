#ifndef VQF_H
#define VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/vqf_block.cuh"


//doesn't need to be explicitly packed
typedef struct vqf {


	uint64_t num_blocks;

	int * locks;


	vqf_block * blocks;

	__device__ void lock_block(uint64_t lock);

	__device__ void unlock_block(uint64_t lock);


	__device__ void lock_blocks(uint64_t lock1, uint64_t lock2);

	__device__ void unlock_blocks(uint64_t lock1, uint64_t lock2);


	__device__ bool insert_empty(uint64_t item);

	__device__ bool insert(uint64_t item);

	__device__ bool query(uint64_t item);

	__device__ bool remove(uint64_t item);



} vqf;




__host__ vqf * build_vqf(uint64_t nitems);


#endif