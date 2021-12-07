#ifndef MEGA_VQF_H
#define MEGA_VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/megablock.cuh"


//doesn't need to be explicitly packed
typedef struct mega_vqf {


	uint64_t num_blocks;

	int * locks;


	megablock * blocks;

	__device__ void lock_block(int warpID, uint64_t lock);

	__device__ void unlock_block(int warpID, uint64_t lock);


	__device__ void lock_blocks(int warpID, uint64_t lock1, uint64_t lock2);

	__device__ void unlock_blocks(int warpId, uint64_t lock1, uint64_t lock2);


	__device__ bool insert(int warpID, uint64_t item);

	__device__ bool query(int warpID, uint64_t item);

	__device__ bool remove(int warpID, uint64_t item);



} mega_vqf;




__host__ mega_vqf * build_mega_vqf(uint64_t nitems);


#endif