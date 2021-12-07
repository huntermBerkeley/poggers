#ifndef _MEGABLOCK_ 
#define _MEGABLOCK_


#include <cuda.h>
#include <cuda_runtime_api.h>






#define TAG_BITS 16


#if TAG_BITS == 8

//need to reconcile # blocks to nums
#define SLOTS_PER_BLOCK 48
#define VIRTUAL_BUCKETS 80

#elif TAG_BITS == 16

#define SLOTS_PER_BLOCK 27
#define VIRTUAL_BUCKETS 36 

#endif



#define LOCK_MASK (1ULL << 63)

#define UNLOCK_MASK ~(1ULL << 63)

typedef struct __attribute__ ((__packed__)) megablock {


	//metadata and tags change based on the size of 
	//tag bits
	uint64_t internal_lock;
	uint64_t counter;
	uint64_t tags[SLOTS_PER_BLOCK];

	__device__ void setup();


	__device__ void lock(int warpID);



	__device__ void unlock(int warpID);


	__device__ int max_capacity();

	
	__device__ int get_fill();


	__device__ void insert(int warpID, uint64_t item);

	__device__ bool query(int warpID, uint64_t item);

	__device__ bool remove(int warpID, uint64_t item);
	//remove related functions



	__device__ bool assert_consistency();


} megablock;



#endif //_MEGABLOCK_