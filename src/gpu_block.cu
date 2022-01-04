
#ifndef _GPU_BLOCK_CU
#define _GPU_BLOCK_CU


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/gpu_block.cuh"
#include "include/warp_utils.cuh"
#include "include/metadata.cuh"

#include <cooperative_groups.h>


//extra stuff
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>

//VQF Block
// Functions Required:

// Lock();
// get_fill();
// Unlock_other();
// Insert();
// Unlock();

//I'm putting the bit manipulation here atm



//set the original 1 bits of the block
//this is done on a per thread level
__device__ void gpu_block::setup(){

	uint64_t mask = UNLOCK_MASK;

	atomicOr((unsigned long long int *) md, mask);

}





// __host__ __device__ static inline int popcntv(const uint64_t val, int ignore)
// {
// 	if (ignore % 64)
// 		return popcnt (val & ~BITMASK(ignore % 64));
// 	else
// 		return popcnt(val);
// }

// Returns the number of 1s up to (and including) the pos'th bit
// Bits are numbered from 0
__host__ __device__ static inline int bitrank(uint64_t val, int pos) {
	val = val & ((2ULL << pos) - 1);
#ifdef __CUDA_ARCH__
	val = __popcll(val);
#else

	//quick fix for summit

	#ifndef __x86_64

		val = __builtin_popcount(val);

	#else

		
		asm("popcnt %[val], %[val]"
			: [val] "+r" (val)
			:
			: "cc");

	#endif
		


#endif
	return val;
}



//make sure these work 
__device__ void gpu_block::lock(int warpID){


	if (warpID == 0){


	uint64_t * data;
	#if TAG_BITS == 8
		data = (uint64_t *) (md + 1);
	#elif TAG_BITS == 16
		data = (uint64_t *) md;
	#endif

	//atomicOr should return 0XXXXXXXXXX - expect the first bit to be zero
	//this means lock_mask & md should be equal to 0 to unlock

	//uint64_t out =	atomicOr((unsigned long long int *) data, LOCK_MASK);

	uint64_t val = atomicOr((unsigned long long int *) data, LOCK_MASK) & LOCK_MASK;


	//uint64_t counter = 0;

	while (val != 0){


		val = atomicOr((unsigned long long int *) data, LOCK_MASK) & LOCK_MASK;

	} 


	}

	//and everyone synchronizes
	__syncwarp();
	

}



//lock local removes the 0 for the lock
// It is needed for other functions such as get_fill().
//
//THIS DOES NOT GRANT EXCLUSIVE ACCESS
// USE ONLY ON SHARED_MEM BLOCKS WITH IMPLIED EXCLUSIVITY
//
__device__ void gpu_block::lock_local(int warpID){

	if (warpID == 0) 


	#if TAG_BITS == 8

	md[1] = md[1] | LOCK_MASK;

	#elif TAG_BITS == 16


	md[0] = md[0] | LOCK_MASK;

	#endif
}


__device__ void gpu_block::unlock(int warpID){

	if (warpID ==0){



	uint64_t * data;
	#if TAG_BITS == 8
		data = (uint64_t *) (md + 1);
	#elif TAG_BITS == 16
		data = (uint64_t *) md;
	#endif

	//double check .ptx on this
	//could cut down cycles a bit if it isn't short
	atomicAnd((unsigned long long int *) data, UNLOCK_MASK);

	}

}


//Adds back the 0 for the lock, acts like unlocking without a flush
// May not be memory safe if used on global blocks
__device__ void gpu_block::unlock_local(int warpID){

	if (warpID == 0) 


	#if TAG_BITS == 8

	md[1] = md[1] & UNLOCK_MASK;

	#elif TAG_BITS == 16


	md[0] = md[0] & UNLOCK_MASK;

	#endif


}





//return the number of filled slots
//much easier to get the number of unfilled slots
//and do capacity - unfilled
//do a trailing zeros count on the rightmost md counter
__device__ int gpu_block::get_fill(){

	#if TAG_BITS == 16
	
	//return SLOTS_PER_BLOCK - __clzll(md[0] & UNLOCK_MASK);

	return 64 - __popcll(md[0]);

	#elif TAG_BITS == 8


	int ones = __popcll(md[1]) + __popcll(md[0]);

	return 128 - ones;

	#endif

	//crash and burn
	// assert(0==1);
	// return 0;

}

__device__ int gpu_block::max_capacity(){

	return SLOTS_PER_BLOCK;
}

//return the index of the ith bit in val 
//logarithmic: this is equivalent to
// mask the first k bits
// such that popcount popcount(val & mask == i);
//return -1 if no such index

//to insert we need to figure out our block
__device__ void gpu_block::insert(int warpID, uint64_t item){

	

	// - slot necessary - the buckets are logical constructs
	//and don't correspond to true indices.

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif

	//md_bit is the metadata index to modify
	//index is the slot we want to insert into the tags
	int fill = get_fill();

	if (warpID == 0){


		tags[fill] = tag;


		md[0] = md[0] << 1;
	}



	__syncwarp();

	return;

}

__device__ bool gpu_block::query(int warpID, uint64_t item){

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	int fill = get_fill();

	int ballot = 0;

	if (warpID < fill){


		if (tags[warpID] == tag) ballot = 1;
	}


	unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);

	int thread_to_query = __ffs(ballot_result) -1;


	if (thread_to_query == -1) return false;

	return true;


}


//attempt to remove an item with tag item, 
//returns false if no item to remove
__device__ bool gpu_block::remove(int warpID, uint64_t item){

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	int fill = get_fill();

	int ballot = 0;

	if (warpID < fill){


		if (tags[warpID] == tag) ballot = 1;
	}


	unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);

	int thread_to_query = __ffs(ballot_result) -1;

	if (thread_to_query == -1) return false;


	bool participating =  (warpID > thread_to_query && warpID < fill);

	if (participating) tag = tags[warpID];

	__syncwarp();

	if (participating) tags[warpID -1] = tag;

	if (warpID == 0){

		#if TAG_BITS == 16
		md[0] = md[0] >> 1;

		#elif TAG_BITS == 8

			you didn't do me yet';å

		#endif
	}

	lock_local(warpID);

	return true;


}


__device__ void gpu_block::bulk_insert(int warpID, uint64_t * items, uint64_t nitems){

	

	// - slot necessary - the buckets are logical constructs
	//and don't correspond to true indices.

	uint64_t item = 0;

	if (warpID < nitems)

		item = items[warpID];

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	//md_bit is the metadata index to modify
	//index is the slot we want to insert into the tags
	int fill = get_fill();

	if (warpID < nitems)

	tags[warpID + fill] = tag;


	//and update metadata
	//its a lot simpler if safety is handled in the vqf
	//so that this doesn't have to do any checks
	if (warpID == 0) 

	#if TAG_BITS == 8
		'YOU DIDNT DO ME YEt';
	#elif TAG_BITS == 16
	
		md[0] = md[0] << nitems;
		
	#endif




	__syncwarp();

	return;

}


__device__ void gpu_block::bulk_insert_team(cooperative_groups::thread_block_tile<32> warpGroup, uint64_t * items, uint64_t nitems){

	#if DEBUG_ASSERTS

	assert (warpGroup.size() == 32);

	#endif

	int warpID = warpGroup.thread_rank();

	// - slot necessary - the buckets are logical constructs
	//and don't correspond to true indices.

	uint64_t item = 0;

	if (warpID < nitems)

		item = items[warpID];

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	// if (warpGroup.thread_rank() != 0){
	// 	printf("Halp: %d %d, %d\n", warpGroup.thread_rank(), tag, nitems);
	// }

	//md_bit is the metadata index to modify
	//index is the slot we want to insert into the tags
	int fill = get_fill();

	if (warpID < nitems)

	tags[warpID + fill] = tag;

	__threadfence();

	warpGroup.sync();

	#if DEBUG_ASSERTS

	if (warpID < nitems){

		// if (warpID != 0){
		// 	assert(tags[warpID + fill-1] != 0);
		// }

		assert (tags[warpID + fill] == tag);


	}


	#endif


	//and update metadata
	//its a lot simpler if safety is handled in the vqf
	//so that this doesn't have to do any checks
	if (warpID == 0) 

	#if TAG_BITS == 8
		'YOU DIDNT DO ME YEt';
	#elif TAG_BITS == 16
	
		md[0] = md[0] << nitems;
		
	#endif




	warpGroup.sync();

	return;

}


__device__ int gpu_block::bulk_query(int warpID, uint64_t * items, uint64_t nitems){

	uint64_t item =0;

	if (warpID < nitems) item = items[warpID];

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif

	int ballot = 0;

	if (warpID < nitems){


		for (int i = 0; i < nitems; i++){
			if (tags[i] == tag) {
				ballot = 1;
				break;
			}
		}
	}
	__syncwarp();

	unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);

	return __popc(ballot_result);



}

__device__ bool gpu_block::assert_consistency(){


	if (get_fill() == __clzll(__brevll(md[0]))){

		return true;

	}

	return false;

}





#endif //GPU_BLOCK_CU