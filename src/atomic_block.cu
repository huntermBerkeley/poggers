
#ifndef _ATOMIC_BLOCK_CU
#define _ATOMIC_BLOCK_CU


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/atomic_block.cuh"
#include "include/warp_utils.cuh"
#include "include/metadata.cuh"
#include "include/sorting_helper.cuh"

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
__device__ void atomic_block::setup(){

	md = 0;

	#if DEBUG_ASSERTS

	//verify that the blocks are aligned according to their idea of a cache line
	assert(sizeof(atomic_block) % BYTES_PER_CACHE_LINE == 0);

	#endif
}






//return the number of filled slots
//much easier to get the number of unfilled slots
//and do capacity - unfilled
//do a trailing zeros count on the rightmost md counter
__device__ int atomic_block::get_fill(){

	return md;

}

__device__ int atomic_block::get_fill_atomic(){

	return atomicCAS((unsigned int * ) &md, (unsigned int) 0, (unsigned int) 0);

}

__device__ int atomic_block::max_capacity(){

	return SLOTS_PER_BLOCK;
}



//atomicAdd to grab the next available slot
__device__ void atomic_block::insert(int warpID, uint64_t item){

	
	if (warpID == 0){
		insert_one_thread(item);
	}
	return;

}

__device__ bool atomic_block::query(int warpID, uint64_t item){


	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;

	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif

	if (tag == TOMBSTONE_VAL){
		tag += 1;
	}

	int fill = get_fill();

//	int fill_cutoff = ((fill -1)/32 + 1) * 32;

	int ballot = 0;


	for (int i = warpID; i < fill; i+=32){

		if (tags[i] == tag) ballot = 1;
	}

	// if (warpID < fill){


	// 	if (tags[warpID] == tag) ballot = 1;
	// }


	unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);

	int thread_to_query = __ffs(ballot_result) -1;


	if (thread_to_query == -1) return false;

	return true;


	// for (int i = warpID; i < fill_cutoff; i+=32){

	// 	if (i < fill && tags[i] == tag) ballot = 1;

	// 	unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);

	// 	int thread_to_query = __ffs(ballot_result) -1;


	// 	if (thread_to_query != -1) return true;
	// }


	// return false;


}


//attempt to remove an item with tag item, 
//returns false if no item to remove
__device__ bool atomic_block::remove(int warpID, uint64_t item){



	//TODO: this


	//return false;


	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif


	int fill = get_fill();

	int ballot = 0;


	//the old one has a problem where multiple items could be deleted

	for (int i =0; i < fill; i+=32){

		int my_slot = i + warpID;


		if (my_slot < fill){
			if (tags[my_slot] == tag) ballot = 1;
		}

		


		unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);

		int thread_to_query = __ffs(ballot_result) -1;

		if (thread_to_query != -1){


			//end
			if (thread_to_query == warpID){

				//change this later
				tags[my_slot] = TOMBSTONE_VAL;
			}

		}




	}


}

__device__ bool atomic_block::purge_tombstone(int warpID){

	int fill = get_fill();


	//keep track of tombstones and non-tombstones, non-tombstones go to the front and tombstones are subtracted from the main list. 
	int non_tombstones =0;

	int tombstones = 0;

	for (int i = warpID; i < fill; i+= 32){

		if (tags[i] == TOMBSTONE_VAL){

			tombstones += 1;
		} else {
			non_tombstones += 1;
		}

	}

	int start= non_tombstones;

	//sync and ballot here
		for (int i=1; i<=16; i*=2) {
	    // We do the __shfl_sync unconditionally so that we
	    // can read even from threads which won't do a
	    // sum, and then conditionally assign the result.
	    int n = __shfl_up_sync(0xffffffff, start, i, 32);
	    if ((warpID) >= i)
	        start += n;
	}

	//read the thread "ahead" of me
	int prev = __shfl_up_sync(0xffffffff, start, 1, 32);

	if (warpID == 0){
		prev = 0;
	} 

	//use prev as start

	#if DEBUG_ASSERTS

	assert(non_tombstones <= 8);

	#endif

	#if TAG_BITS == 16 

	uint16_t temp_bits[8];

	int temp_start = 0;

	#else 

	uint8_t temp_bits[8];

	int temp_start = 0;

	#endif

	for (int i = warpID; i < fill; i++){

		if (tags[i] != TOMBSTONE_VAL){

			temp_bits[temp_start] = tags[i];
			temp_start+=1;
		}

	}

	//regroup for write
	__syncwarp();

	atomicSub((unsigned int *) & md, (unsigned int) tombstones);

	for (int i=0; i < temp_start; i++){

		tags[prev + i] = temp_bits[i];
	}



}


__device__ bool atomic_block::insert_one_thread(uint64_t item){

		#if TAG_BITS == 8
			uint8_t tag = item & 0xFF;
		#elif TAG_BITS == 16
			uint16_t tag = item & 0xFFFF;
		#endif

		if (tag == TOMBSTONE_VAL){
			tag += 1;
		}

		int fill = atomicAdd((unsigned int *) &md, (unsigned int) 1);

		if (fill < SLOTS_PER_BLOCK){
			tags[fill] = tag;
		} else {
			//undo addition so that removes function as expected
			atomicSub((unsigned int *) & md, (unsigned int) 1);
			return false;
		}

		

	__threadfence();


	return true;
}


__device__ void atomic_block::bulk_insert(int warpID, uint64_t * items, uint64_t nitems){

	

	// - slot necessary - the buckets are logical constructs
	//and don't correspond to true indices.


	//for the 16 bit 64 byte case maybe write a preprocessor directive to not do the loup

	int fill = get_fill();

	for (int i = warpID; i < nitems; i+=32){

		uint64_t item = items[i];


		#if TAG_BITS == 8
			uint8_t tag = item & 0xFF;
		#elif TAG_BITS == 16
			uint16_t tag = item & 0xFFFF;
		#endif

		if (tag == TOMBSTONE_VAL){
			tag += 1;
		}


		tags[i + fill] = tag;
		

	}
	


	if (warpID == 0) atomicAdd((unsigned int *) & md, nitems);


	__syncwarp();

	return;

}


__device__ void atomic_block::sorted_bulk_insert(uint8_t * temp_tags, uint64_t * items, uint64_t nitems, int teamID, int warpID){
	

	
	//for the 16 bit 64 byte case maybe write a preprocessor directive to not do the loop



	int fill = get_fill();


	//without debug on you can mess this up, safety checks are handled at that level by higher up
	//processes
	#if DEBUG_ASSERTS


	assert(byte_assert_sorted(items, nitems));

	assert(short_byte_assert_sorted(tags, fill));


	#endif



	//now that bounds are checked, setup for main insert

	merge_dual_arrays_8_bit_64_bit(temp_tags, &tags[0], items, fill, nitems, teamID, warpID);


	


	if (warpID == 0) atomicAdd((unsigned int *) & md, nitems);



	__syncwarp();


	#if DEBUG_ASSERTS



	if (!short_byte_assert_sorted(tags, fill+nitems)){

		assert(short_byte_assert_sorted(tags, fill+nitems));

	}


	#endif

	return;

}


//a variant of the insert scheme that treats temp_tags as a local array, because it is
//this is a workaround to redefining the shared memory structure of the entire project
// while still maintaining minimal memory use
__device__ void atomic_block::sorted_bulk_finish(uint8_t * temp_tags, uint8_t * items, uint64_t nitems, int teamID, int warpID){


	int fill = get_fill();

	#if DEBUG_ASSERTS

	assert(short_byte_assert_sorted(items, nitems));

	assert(short_byte_assert_sorted(tags, fill));


	if (nitems + fill >= SLOTS_PER_BLOCK){

		assert(nitems + fill <= SLOTS_PER_BLOCK);
	}
	


	#endif


	//now that bounds are checked, setup for main insert

	//TODO fix merge_dual_arrays
	merge_dual_arrays(temp_tags, &tags[0], items, fill, nitems, teamID, warpID);


	


	if (warpID == 0) atomicAdd((unsigned int *) & md, nitems);



	__syncwarp();


	#if DEBUG_ASSERTS



	if (!short_byte_assert_sorted(tags, fill+nitems)){

		assert(short_byte_assert_sorted(tags, fill+nitems));

	}


	#endif

	return;







}



//TODO: Patch this
//BUlk Query can only find items that are < 32
__device__ int atomic_block::bulk_query(int warpID, uint64_t * items, uint64_t nitems){

	#if DEBUG_ASSERTS

	assert(nitems < 32);

	#endif

	uint64_t item =0;

	if (warpID < nitems) item = items[warpID];

	#if TAG_BITS == 8
		uint8_t tag = item & 0xFF;
	#elif TAG_BITS == 16
		uint16_t tag = item & 0xFFFF;
	#endif

	if (tag == TOMBSTONE_VAL){
		tag += 1;
	}

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

__device__ bool atomic_block::assert_consistency(){


	if (md <= SLOTS_PER_BLOCK) return true;

	return false;

}





//replace this with a recursive bitonic sort
__device__ bool atomic_block::sort_block(int teamID, int warpID){


	// int fill = get_fill();

	// shortByteBitonicSort(tags, 0, fill, true, warpID);

	// __syncwarp();

	int fill = get_fill();

	short_warp_sort(tags, fill, teamID, warpID);

	//bubble_sort(tags, fill, warpID);


	// while (true){


	// 	bool sorted = false;

	// 	//even transpositions
	// 	for (int i = warpID*2+1; i < fill; i+=64){

	// 		//swap warpID*2, warpID*2+1

	// 		if ((tags[i-1] & 0xFF) > (tags[i] & 0xFF)){

	// 			#if TAG_BITS == 8

	// 			uint8_t temp_tag;

	// 			#else

	// 			uint16_t temp_tag;

	// 			#endif

	// 			temp_tag = tags[i-1];

	// 			tags[i-1] = tags[i];

	// 			tags[i] = temp_tag;

	// 			sorted = true;

	// 		}



	// 	}


	// 	//odd transpositions
	// 	for (int i = warpID*2+2; i < fill; i+=64){

	// 		//swap warpID*2, warpID*2+1

	// 		if ((tags[i-1] & 0xFF) > (tags[i] & 0xFF)){

	// 			#if TAG_BITS == 8

	// 			uint8_t temp_tag;

	// 			#else

	// 			uint16_t temp_tag;

	// 			#endif

	// 			temp_tag = tags[i-1];

	// 			tags[i-1] = tags[i];

	// 			tags[i] = temp_tag;

	// 			sorted = true;

	// 		}



	// 	}

	// 	if (__ffs(__ballot_sync(0xffffffff, sorted)) == 0) return;


	// }


}


//this is a check, no fancy schmancyness
__device__ bool atomic_block::assert_sorted(int warpID){


	int fill = get_fill();

	return short_byte_assert_sorted(tags, fill);

}



//inner sorted join
//assume both are prepped and sorted
//ill fix the comparison shit later
__device__ bool atomic_block::sorted_bulk_query(int warpID, uint64_t * items, bool * found, uint64_t nitems){


	//byteBitonicSort(items, 0, nitems, true, warpID);


	//big_bubble_sort(items, nitems, warpID);

	assert(byte_assert_sorted(items, nitems));

	//bitonicSort(uint64_t * items, int low, int count, bool dir, int warpID){

	int fill = get_fill();

	//bubble_sort(tags, fill, warpID);

	__syncwarp();

	assert(short_byte_assert_sorted(tags, fill));



	int left = 0;
	int right = 0;

	while (true){

		#if TAG_BITS == 8

		uint8_t comp = items[left] & 0xFF;

		#else

		uint16_t comp = items[left] & 0xFFFF;

		#endif

		if (comp == tags[right]){

			found[left] = true;
			left++;

			if (left >= nitems) return;


		} else if (comp < tags[right]){

			//left is a miss
			found[left] = false;
			left++;

			if (left >= nitems) return;

		} //else if (items[left] > tags[right])
		else {

			right++;

			if (right >= fill){

				//purge remaining 
				for (int i = left; i < nitems; i++){

					found[i] = false;

				}

				return;

			}

		

		}




	}



} 



__device__ bool atomic_block::binary_search_query(uint64_t item){

	int fill = get_fill();

	#if DEBUG_ASSERTS

	assert(short_byte_assert_sorted(tags, fill));

	#endif


	#if TAG_BITS == 8

	uint8_t tag = item & 0xff;

	#endif


	int lower = 0;

	int upper = fill;

	int index;


	while (upper != lower){

		index = lower + (upper - lower)/2;


		int query_item = tags[index];

		if (query_item < tag){

			lower = index+1;

		} else if (query_item > tag){

			upper = index;

		} else {

			return true;
		}


	}

	if (lower < fill && tags[lower] == tag) return true;

	return false;





}

#endif //atomic_block_CU