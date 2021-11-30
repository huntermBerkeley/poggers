


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/vqf_team_block.cuh"
#include "include/warp_utils.cuh"


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

__host__ __device__ static inline int popcnt(uint64_t val)
{
#ifdef __CUDA_ARCH__
	val = __popcll(val);
#else

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


//set the original 1 bits of the block
//this is done on a per thread level
__device__ void vqf_block::setup(){

	uint64_t mask = (1ULL << VIRTUAL_BUCKETS)-1;

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
__device__ void vqf_block::lock(int warpID){


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


__device__ void vqf_block::unlock(int warpID){

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



//shifts bits above cutoff to the left one
//leaves bits below untouched
//the new bit at cutoff is a 0
//as this is the default for inserts
__device__ uint64_t vqf_block::shift_upper_bits(uint64_t bits, int cutoff){

	//use masking table here
	//uint64_t mask = (1ULL << (cutoff +1)) -1;
	uint64_t mask = (1ULL << (cutoff)) -1;

	uint64_t lower = bits & mask;

	uint64_t upper = bits & (~mask);


	return lower + upper + upper;

	//alt: return lower + (upper << 1)


}

//uint 64_t shifts bits above the cutoff to the left one
//leaves bits below untouched
//assumes that the bits[cutoff] == 0
// undefined if that's not true. It should always be 0 as only inserted items
// can be removed
__device__ uint64_t vqf_block::shift_lower_bits(uint64_t bits, int cutoff){

	uint64_t mask = (1ULL << (cutoff)) -1;

	uint64_t lower = bits & mask;

	uint64_t upper = bits & (~mask);

	return lower + (upper >> 1);

}


__device__ uint64_t vqf_block::get_upper_bit(uint64_t bits){

	return bits & LOCK_MASK;
}


__device__ uint64_t vqf_block::get_lower_bit(uint64_t bits){
	return bits & 1ULL;
}

//Given an index, insert a 0 into the slot, shifting all metadata
//one to the right
__device__ void vqf_block::md_0_and_shift_right(int index){


	//8 Bit case: pretty simple shift
	
	#if TAG_BITS == 8

	//check if index is too large for 0

	if (index >= 64){


		index = index - 64;


		uint64_t lock = get_upper_bit(md[1]);

		uint64_t new_md = shift_upper_bits(md[1], index);

		atomicExch(((unsigned long long int *) md) + 1, new_md | lock);



	} else {


		//clear space in the upper bits

		uint64_t lock = get_upper_bit(md[1]);

		uint64_t bit_lower = get_upper_bit(md[0]);

		uint64_t new_upper = shift_upper_bits(md[1], 0);

		//replace upper bits so we don't have to think about them
		atomicExch(((unsigned long long int *)md) + 1, new_upper | lock | __brevll(bit_lower));

		uint64_t new_lower = shift_upper_bits(md[0], index);

		atomicExch((unsigned long long int *)md , new_lower);


	}


	#elif TAG_BITS == 16

	//simple case - one vector for all md
	uint64_t lock = get_upper_bit(md[0]);

	uint64_t new_md = shift_upper_bits(md[0], index);

	atomicExch((unsigned long long int *)md, new_md | lock);


	#endif


	return;
}

//Given an index, shifting all metadata
//one to the left, eliminating the slot
__device__ void vqf_block::down_shift(int index){


	//8 Bit case: pretty simple shift
	
	#if TAG_BITS == 8

	//check if index is too large for 0

	if (index >= 64){


		index = index - 64;


		uint64_t lock = get_upper_bit(md[1]);



		uint64_t new_md = shift_lower_bits(md[1] & UNLOCK_MASK, index);

		atomicExch(((unsigned long long int *) md) + 1, new_md | lock);



	} else {


		//clear space in the upper bits

		uint64_t lock = get_upper_bit(md[1]);

		uint64_t moved_bit = get_lower_bit(md[0]);

		//shuffle down the upper bits, and pass over the last bit
		uint64_t new_upper = shift_lower_bits(md[1] & UNLOCK_MASK, 0);

		//replace upper bits so we don't have to think about them
		atomicExch(((unsigned long long int *)md) + 1, new_upper | lock );

		uint64_t new_lower = shift_lower_bits(md[0], index);

		//replace topmost bit with new bit
		atomicExch((unsigned long long int *)md , new_lower | __brevll(moved_bit));


	}


	#elif TAG_BITS == 16

	//simple case - one vector for all md
	uint64_t lock = get_upper_bit(md[0]);

	uint64_t new_md = shift_lower_bits(md[0] & UNLOCK_MASK, index);

	atomicExch((unsigned long long int *)md, new_md | lock);


	#endif


	return;
}



//return the number of filled slots
//much easier to get the number of unfilled slots
//and do capacity - unfilled
//do a trailing zeros count on the rightmost md counter


__device__ int vqf_block::get_fill(){

	#if TAG_BITS == 16
	
	return SLOTS_PER_BLOCK - __clzll(md[0] & UNLOCK_MASK);


	#elif TAG_BITS == 8

	int upper_count = __clzll(md[1] & UNLOCK_MASK);

	if (upper_count == 64){

		upper_count += __clzll(md[0]);

	}

	return SLOTS_PER_BLOCK - upper_count;

	#endif

	//crash and burn
	// assert(0==1);
	// return 0;

}

__device__ int vqf_block::max_capacity(){

	return SLOTS_PER_BLOCK;
}

//return the index of the ith bit in val 
//logarithmic: this is equivalent to
// mask the first k bits
// such that popcount popcount(val & mask == i);
//return -1 if no such index

//to insert we need to figure out our block
__device__ void vqf_block::insert(int warpID, uint64_t item){

	int slot = item % VIRTUAL_BUCKETS;

	// - slot necessary - the buckets are logical constructs
	//and don't correspond to true indices.


	//md_bit is the metadata index to modify
	//index is the slot we want to insert into the tags
	int md_bit = warp_utils::select(warpID, md[0], slot);


	if (md_bit == -1){

		printf("%d Metadata %llu\n", warpID, md[0]);
		printf("%d slot %d\n", warpID, slot);

	}
	assert(md_bit != -1);




	int index = md_bit - slot;

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	//index is the slot to insert into

	//there are get_fill slots in use

	int num_slots_to_shift = get_fill() - index;

	//we are starting at index,

	//there are 

	//dest, src, slots

	#if TAG_BITS == 16

	warp_utils::block_8_memmove_insert(warpID, tags, tag, index);


	if (warpID == 0)
	md_0_and_shift_right(md_bit);


	__syncwarp();

	#else 

	warp_utils::warp_memmove(warpID, tags+index+1, tags+index, num_slots_to_shift*sizeof(tags[0]));

	if (warpID == 0){
	tags[index] = tag;

	md_0_and_shift_right(md_bit);

	}

	__syncwarp();

	#endif

	
	//push items up and over

	__threadfence();

	//sync after fence so we know all warps are done writing
	__syncwarp();


	



}

__device__ bool vqf_block::query(int warpID, uint64_t item){

	int slot = item % VIRTUAL_BUCKETS;

	int start  = warp_utils::select(warpID, md[0], slot-1) - (slot -1);
	if (slot == 0) start = 0;
	

	int end = warp_utils::select(warpID, md[0], slot) - slot;

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	int ballot = 0;
	for (int i=start+warpID; i < end; i+=32){

		if (tags[i] == tag) ballot = 1;
	}
	__syncwarp();

	unsigned int ballot_result = __ballot_sync(0xfffffff, ballot);


	//if ballot is 0 no one found

	return !(ballot_result == 0);

}


//attempt to remove an item with tag item, 
//returns false if no item to remove
__device__ bool vqf_block::remove(int warpID, uint64_t item){

	int slot = item % VIRTUAL_BUCKETS;

	int start = warp_utils::select(warpID, md[0], slot-1) - (slot -1);
	if (slot == 0) start = 0;
	

	int end = warp_utils::select(warpID, md[0], slot) - slot;

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	int ballot = 0;

	int insert_index = 0;
	for (int i=start+warpID; i < end; i+=32){

		if (tags[i] == tag){
			ballot = 1;
			insert_index = i;
		}


	}


	//synchronize based on ballot, propogate starting index

	unsigned int ballot_result = __ballot_sync(0xfffffff, ballot);


	int thread_to_query = __ffs(ballot_result)-1;

	//Could not remove!
	if (thread_to_query == -1) return false;


	int remove_index = __shfl_sync(0xfffffff, insert_index, thread_to_query); 

	int num_slots_to_shift = get_fill() - remove_index;


	//do the reverse move


	#if TAG_BITS == 16 


	warp_utils::block_8_memmove_remove(warpID, tags, remove_index);

	#else

	warp_utils::warp_memmove(warpID, tags+remove_index, tags+remove_index+1, num_slots_to_shift*sizeof(tags[0]));

	//tags are shrunk, now change metadata
	

	#endif

	if (warpID == 0){

		down_shift(remove_index+slot);
	}

	__syncwarp();
	__threadfence();
	__syncwarp();

	return true;

}


__device__ void vqf_block::printBlock(){

	printf("Metadata %llu\n", md);

	printf("Fill: %d\n", get_fill());

	// printf("Tags:\n");

	// for (int i =0; i < SLOTS_PER_BLOCK; i+=10){


	// 	for (int j = 0; j <10; j++){

	// 		if (i+j < SLOTS_PER_BLOCK){
	// 			printf("%X ", tags[i+j]);
	// 		}

	// 	}

	// 	printf("\n");




	// }


}


__device__ void vqf_block::printMetadata(){

	for (uint64_t i = 63; i <= 0; i--){


		printf("%d", md[0] & (1<<i));
	}
	printf("\n");
}


__device__ bool vqf_block::assert_consistency(){


	return(popcnt(md[0]) == VIRTUAL_BUCKETS + 1);
}
