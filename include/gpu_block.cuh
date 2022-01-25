#ifndef _GPU_BLOCK_ 
#define _GPU_BLOCK_


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cooperative_groups.h>




#define TAG_BITS 16


#if TAG_BITS == 8

//need to reconcile # blocks to nums
#define SLOTS_PER_BLOCK 48
#define VIRTUAL_BUCKETS 80

#elif TAG_BITS == 16

#define SLOTS_PER_BLOCK 28
#define VIRTUAL_BUCKETS 36 

#endif



#define LOCK_MASK (1ULL << 63)

#define UNLOCK_MASK ~(1ULL << 63)

//POSSIBLE BUG:
//the volatile status of md is what causes the slow performance

typedef struct __attribute__ ((__packed__)) gpu_block {


	//metadata and tags change based on the size of 
	//tag bits
	#if TAG_BITS == 8

		volatile uint64_t md[2];
		uint8_t tags[48];

	#elif TAG_BITS == 16

		
		uint64_t md[1];
		uint16_t tags[28];

	#endif

	__device__ void setup();


	__device__ void lock_one_thread();

	__device__ void unlock_one_thread();

	__device__ bool insert_one_thread(uint64_t item);


	__device__ void lock(int warpID);


	__device__ void lock_local(int warpID);




	__device__ void unlock(int warpID);


	__device__ void unlock_local(int warpID);


	__device__ int max_capacity();

	__device__ int get_fill();


	__device__ void insert(int warpID, uint64_t item);

	__device__ bool query(int warpID, uint64_t item);

	__device__ bool remove(int warpID, uint64_t item);


	__device__ void bulk_insert(int warpID, uint64_t * items, uint64_t nitems);

	__device__ void bulk_insert_team(cooperative_groups::thread_block_tile<32> warpGroup, uint64_t * items, uint64_t nitems);


    __device__ int bulk_query(int warpID, uint64_t * items, uint64_t nitems);


	__device__ bool assert_consistency();
	

} gpu_block;

// #if TAG_BITS == 8
// 	// We are using 8-bit tags.
// 	// One block consists of 48 8-bit slots covering 80 buckets, and 80+48 = 128
// 	// bits of metadata.
// 	typedef struct __attribute__ ((__packed__)) vqf_block {
// 		uint64_t md[2];
// 		uint8_t tags[48];

// 		void test();
// 	} 
// #elif TAG_BITS == 12
// 	// We are using 12-bit tags.
// 	// One block consists of 32 12-bit slots covering 96 buckets, and 96+32 = 128
// 	// bits of metadata.
//         // NOTE: not supported yet.
// 	typedef struct __attribute__ ((__packed__)) vqf_block {
// 		uint64_t md[2];
// 		uint8_t tags[32]; // 32 12-bit tags
  
// 		void test();
// 	} vqf_block;
// #elif TAG_BITS == 16 
// 	// We are using 16-bit tags.
// 	// One block consists of 28 16-bit slots covering 36 buckets, and 36+28 = 64
// 	// bits of metadata.
// 	typedef struct __attribute__ ((__packed__)) vqf_block {
// 		uint64_t md;
// 		uint16_t tags[28];

// 		void test();
// 	} vqf_block;
// #endif


//DEFINE FUNCS

__device__ bool compare_blocks(gpu_block one, gpu_block two);



#endif //GPU_BLOCK_