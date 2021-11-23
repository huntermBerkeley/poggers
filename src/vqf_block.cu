


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/vqf_block.cuh"


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

//a variant of memmove that compares the two pointers
__device__ void gpu_memmove(void* dst, const void* src, size_t n)
{
	//printf("Launching memmove\n");
	//todo: allocate space per thread for this buffer before launching the kernel

	char * char_dst = (char *) dst;
	char * char_src = (char *) src;

  //double check this,
  //think it is just > since dst+n does not get copied

  

  if (char_src+n > char_dst && char_src < char_dst){

  	//copy backwards 
  	for (int i =n-1; i >= 0; i--){



  		char_dst[i] = char_src[i];

  	}

  } else {

  	//copy regular
  	for (int i =0; i<n; i++){
  		char_dst[i] = char_src[i];
  	}


  }

  //free(temp_buffer);

}

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
__device__ void vqf_block::lock(){

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


__device__ void vqf_block::extra_lock(uint64_t block_index){

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


//	uint64_t counter = 0;

	while (val != 0){


		// counter += 1;


		// if (counter > 10000000){

		// 	//printBlock();

		// 	printf("Block broke at Block index: %llu\n", block_index);

		// 	return;
		// }

		val = atomicOr((unsigned long long int *) data, LOCK_MASK) & LOCK_MASK;

	} 

}

__device__ void vqf_block::unlock(){

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
	
	return VIRTUAL_BUCKETS - __clzll(md[0] & UNLOCK_MASK);


	#elif TAG_BITS == 8

	int upper_count = __clzll(md[1] & UNLOCK_MASK);

	if (upper_count == 64){

		upper_count += __clzll(md[0]);

	}

	return VIRTUAL_BUCKETS - upper_count;

	#endif

	//crash and burn
	// assert(0==1);
	// return 0;

}

__device__ int vqf_block::max_capacity(){

	return VIRTUAL_BUCKETS;
}

//return the index of the ith bit in val 
//logarithmic: this is equivalent to
// mask the first k bits
// such that popcount popcount(val & mask == i);
//return -1 if no such index
__device__ int vqf_block::select(volatile uint64_t* val_arr, int bit){


	//slow version
	//I can do this with a precompute table
	//should save cycles?
	uint64_t val = val_arr[0];

	#if TAG_BITS == 8

	//need to check which metadata bit we're looking at

	int offset = 0;

	if (popcnt(val_arr[0]) < bit) {

		val = val_arr[0];
		offset = 0;

	} else {

		val = val_arr[1];
		offset = 64;
	}

	#endif




	for (int i=0; i< bit; i++){
		val = val & (val-1);
	}

	//prolly need a ffsll here

	#if TAG_BITS == 8

	uint64_t intermediate = val & ~(val-1);

	return __ffsll(intermediate)+offset -1;

	#endif 

	uint64_t intermediate = val & ~(val-1);

	return __ffsll(intermediate) -1;

}

//to insert we need to figure out our block
__device__ void vqf_block::insert(uint64_t item){

	int slot = item % VIRTUAL_BUCKETS;

	// - slot necessary - the buckets are logical constructs
	//and don't correspond to true indices.
	int index = select(md, slot) - slot;

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
	gpu_memmove(tags+index+1, tags+index, num_slots_to_shift*sizeof(tags[0]));
	//push items up and over
	tags[index] = tag;

	md_0_and_shift_right(index + slot);

	__threadfence();



}

__device__ bool vqf_block::query(uint64_t item){

	int slot = item % VIRTUAL_BUCKETS;

	//uint64_t md_old = md[0];


	int start  = select(md, slot-1) - (slot -1);
	if (slot == 0) start = 0;
	

	int end = select(md, slot) - slot;

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif

	//assert(md_old == md[0]);

	for (int i=start; i < end; i++){

		if (tags[i] == tag) return true;
	}



	return false;

}


//attempt to remove an item with tag item, 
//returns false if no item to remove
__device__ bool vqf_block::remove(uint64_t item){

	int slot = item % VIRTUAL_BUCKETS;

	int start  = select(md, slot-1) - (slot -1);
	if (slot == 0) start = 0;
	

	int end = select(md, slot) - slot;

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif

	for (int i=start; i < end; i++){

		if (tags[i] == tag){

			//we have an item!

			//shift the slots back


			//rewrite the metadata
			int num_slots_to_shift = get_fill() - i;

			gpu_memmove(tags+i, tags+i+1, num_slots_to_shift*sizeof(tags[0]));

			down_shift(i+slot);
			__threadfence();

			//return item cleaned
			//since this cuts the loop, it should be memory safe
			return true;

		} 
	}

	return false;

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
