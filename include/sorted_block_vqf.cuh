#ifndef SORTED_BLOCK_VQF_H
#define SORTED_BLOCK_VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/atomic_block.cuh"

#include "include/metadata.cuh"


typedef struct __attribute__ ((__packed__)) thread_team_block {


	atomic_block internal_blocks[BLOCKS_PER_THREAD_BLOCK];

} thread_team_block;


//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) optimized_vqf {




	uint64_t num_blocks;

	uint64_t num_teams;
	
	uint64_t ** buffers;

	uint64_t * buffer_sizes;

	thread_team_block * blocks;

	int seed;

	__device__ void lock_block(int warpID, uint64_t team, uint64_t lock);

	__device__ void unlock_block(int warpID, uint64_t team, uint64_t lock);


	__device__ void lock_blocks(int warpID, uint64_t team1, uint64_t lock1, uint64_t team2, uint64_t lock2);

	__device__ void unlock_blocks(int warpId, uint64_t team1, uint64_t lock1, uint64_t team2, uint64_t lock2);


	__host__ void bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses);


	//insert while maintaining a sorted order inside of buckets
	__host__ void sorted_bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses);

	
	//block variants for debugging

	__device__ bool mini_filter_block(uint64_t * misses);

	__device__ bool sorted_mini_filter_block(uint64_t * misses);

	__device__ void dump_remaining_buffers_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID, uint64_t * misses);

    __device__ bool insert_single_buffer_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID);


	__device__ bool query(int warpID, uint64_t key);

	//query but we check both spots - slower
	__device__ bool full_query(int warpID, uint64_t key);

	//__device__ bool remove(int warpID, uint64_t item);

	//TODO: make this meaningfully distinct from the sorted variation - currently sorted
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

	__device__ bool query_single_buffer_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID, uint64_t * items, bool * hits);

	__device__ bool mini_filter_queries(uint64_t * items, bool * hits);

	__host__ void bulk_query(uint64_t * items, uint64_t nitems, bool * hits);

	__host__ void insert_async(uint64_t * items, uint64_t nitems, uint64_t num_teams, uint64_t num_blocks, cudaStream_t stream, uint64_t * misses);

	__host__ void sort_and_check();

	__device__ bool mini_filter_bulk_queries(uint64_t * items, bool * hits);

	__host__ void sorted_bulk_query(uint64_t * items, uint64_t nitems, bool * hits);


	//sliced inserts for timing
	__host__ void sorted_bulk_insert_buffers_preattached(uint64_t * misses);


	//functions for working on the single write variant
	__device__ bool buffer_get_primary_count(thread_team_block * local_blocks, int * counters, uint64_t blockID, int warpID, int block_warpID, int threadID);



	//extra stuff for async write
	__device__ bool sorted_mini_filter_block_async_write(uint64_t * misses);

	//__device__ bool buffer_get_primary_count(thread_team_block * local_blocks, int * counters, uint64_t blockID, int warpID, int block_warpID, int threadID);


	__device__ void dump_all_buffers_sorted(thread_team_block * local_blocks, int * offsets, int * counters, uint64_t blockID, int warpID, int threadID, uint64_t * misses);


	__device__ bool query_single_item_sorted_debug(int warpID, uint64_t hash);


	//internal descriptions
	__host__ void get_average_fill_block();

	__host__ void get_average_fill_team();



	#if TAG_BITS == 8

		__device__ bool sorted_insert_single_buffer_block(thread_team_block * local_blocks, uint8_t * temp_tags, uint64_t blockID, int warpID, int block_warpID, int threadID);

		__device__ void dump_remaining_buffers_sorted(thread_team_block * local_blocks, uint8_t * temp_tags, uint64_t blockID, int warpID, int threadID, uint64_t * misses);



	#elif TAG_BITS == 16

		__device__ bool sorted_insert_single_buffer_block(thread_team_block * local_blocks, uint16_t * temp_tags, uint64_t blockID, int warpID, int block_warpID, int threadID);

		__device__ void dump_remaining_buffers_sorted(thread_team_block * local_blocks, uint16_t * temp_tags, uint64_t blockID, int warpID, int threadID, uint64_t * misses);


	#endif

} optimized_vqf;


__host__ optimized_vqf * prep_host_vqf(uint64_t nitems);


__host__ optimized_vqf * build_vqf(uint64_t nitems);


#endif