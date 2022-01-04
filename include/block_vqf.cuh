#ifndef BLOCK_VQF_H
#define BLOCK_VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/gpu_block.cuh"

#include "include/metadata.cuh"

#include <cooperative_groups.h>


typedef struct __attribute__ ((__packed__)) thread_team_block {


	gpu_block internal_blocks[WARPS_PER_BLOCK];

} thread_team_block;


//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) optimized_vqf {


	uint64_t num_blocks;

	uint64_t num_teams;
	
	uint64_t ** buffers;

	uint64_t * buffer_sizes;

	int * locks;

	thread_team_block * blocks;

	int seed;

	__device__ void lock_block(int warpID, uint64_t team, uint64_t lock);

	__device__ void unlock_block(int warpID, uint64_t team, uint64_t lock);


	__device__ void lock_blocks(int warpID, uint64_t team1, uint64_t lock1, uint64_t team2, uint64_t lock2);

	__device__ void unlock_blocks(int warpId, uint64_t team1, uint64_t lock1, uint64_t team2, uint64_t lock2);


	__host__ void bulk_insert(uint64_t * items, uint64_t nitems);

	__device__ bool mini_filter_insert(uint64_t teamID);

	__device__ bool insert_single_buffer(cooperative_groups::thread_block_tile<32> warpGroup, thread_team_block * local_blocks, uint64_t teamID, uint64_t buffer);


	__device__ bool query(int warpID, uint64_t key);

	//query but we check both spots - slower
	//__device__ bool full_query(int warpID, uint64_t key);

	//__device__ bool remove(int warpID, uint64_t item);

	__host__ void attach_buffers(uint64_t * vals, uint64_t nvals);

	//__global__ void set_buffers_binary(uint64_t num_keys, uint64_t * keys);

	//__global__ void set_buffer_lens(uint64_t num_keys, uint64_t * keys);


	__device__ uint64_t hash_key(uint64_t key);

	__device__ bool buffer_insert(int warpID, uint64_t buffer);

	__device__ int buffer_query(int warpID, uint64_t buffer);

	__host__ uint64_t get_num_buffers();

	__host__ uint64_t get_num_teams();

	__device__ uint64_t get_bucket_from_hash(uint64_t hash);

	__device__ uint64_t get_alt_hash(uint64_t hash, uint64_t bucket);

	//__device__ bool shared_buffer_insert(int warpID, int shared_blockID, uint64_t buffer);

	//__device__ bool shared_buffer_insert_check(int warpID, int shared_blockID, uint64_t buffer);

	//__device__ bool multi_buffer_insert(int warpID, int shared_blockID, uint64_t start_buffer);

	__device__ void multi_buffer_insert(int warpID, int init_blockID, uint64_t start_buffer);


	//power of two choice functions 
	__host__ void insert_power_of_two(uint64_t * vals, uint64_t nitems);


} optimized_vqf;




__host__ optimized_vqf * build_vqf(uint64_t nitems);


#endif