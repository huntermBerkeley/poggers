#ifndef _ATOMIC_BLOCK_H 
#define _ATOMIC_BLOCK_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/metadata.cuh"


#define BYTES_AVAILABLE (BYTES_PER_CACHE_LINE * CACHE_LINES_PER_BLOCK - sizeof(unsigned int))

//the best performance comes with block dumps that are very large
//so these blocks can be configured to allow for much larger inserts

#if TAG_BITS == 8

//need to reconcile # blocks to nums


#define SLOTS_PER_BLOCK (BYTES_AVAILABLE)



#elif TAG_BITS == 16

#define SLOTS_PER_BLOCK (BYTES_AVAILABLE / 2) 


#endif


#define VIRTUAL_BUCKETS (5 * SLOTS_PER_BLOCK / 4)
#define FILL_CUTOFF (3 * SLOTS_PER_BLOCK / 4)

//POSSIBLE BUG:
//the volatile status of md is what causes the slow performance
//this was right, global cache calls fuck it all up so the new versions don't do locking
//these features are experimental and might not work
//this has a much weaker precondition

typedef struct __attribute__ ((__packed__)) atomic_block {


	//tag bits change based on the #of bytes allocated per block


	unsigned int md;



	#if TAG_BITS == 8

		uint8_t tags[SLOTS_PER_BLOCK];

	#elif TAG_BITS == 16

		
		uint16_t tags[SLOTS_PER_BLOCK];

	#endif

	__device__ void setup();


	__device__ int get_fill();


	__device__ int max_capacity();



	//use a failed CAS to get a better estimate
	__device__ int get_fill_atomic();



	__device__ bool insert_one_thread(uint64_t item);

	__device__ void insert(int warpID, uint64_t item);

	__device__ bool query(int warpID, uint64_t item);

	__device__ void bulk_insert(int warpID, uint64_t * items, uint64_t nitems);

    __device__ int bulk_query(int warpID, uint64_t * items, uint64_t nitems);


    //remove specific code

    __device__ bool remove(int warpID, uint64_t item);
    __device__ bool purge_tombstone(int warpID);


	__device__ bool assert_consistency();
	

} atomic_block;

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




#endif //GPU_BLOCK_