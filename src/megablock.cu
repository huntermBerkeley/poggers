


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/megablock.cuh"
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

//set the original 1 bits of the block
//this is done on a per thread level
__device__ void megablock::setup(){

	internal_lock = 0;
	counter = 0; 

}



//make sure these work 
__device__ void megablock::lock(int warpID){


	if (warpID == 0){

		while(atomicCAS((unsigned long long int *)& internal_lock, 0,1) != 0);

	}

	//and everyone synchronizes
	__syncwarp();
	

}


__device__ void megablock::unlock(int warpID){

	if (warpID ==0){


		while(atomicCAS((unsigned long long int *)& internal_lock, 1, 0) != 1);
		
	}

	__syncwarp();

}




__device__ int megablock::get_fill(){

	return counter;

}

__device__ int megablock::max_capacity(){

	return SLOTS_PER_BLOCK;
}

//return the index of the ith bit in val 
//logarithmic: this is equivalent to
// mask the first k bits
// such that popcount popcount(val & mask == i);
//return -1 if no such index

//to insert we need to figure out our block
__device__ void megablock::insert(int warpID, uint64_t item){

	
	int count = counter;
	//push items up and over

	if (warpID == 0){

		tags[counter] = item;

		counter+=1;


	}


	__threadfence();

	//sync after fence so we know all warps are done writing
	__syncwarp();


	



}

__device__ bool megablock::query(int warpID, uint64_t item){


	int	ballot = (warpID < counter && (tags[warpID] == item));

	__syncwarp();




	unsigned int ballot_result = __ballot_sync(0xfffffff, ballot);


	//if ballot is 0 no one found

	return !(ballot_result == 0);

}


//attempt to remove an item with tag item, 
//returns false if no item to remove
__device__ bool megablock::remove(int warpID, uint64_t item){

	int	ballot = (warpID < counter && (tags[warpID] == item));

	//synchronize based on ballot, propogate starting index

	unsigned int ballot_result = __ballot_sync(0xfffffff, ballot);


	int thread_to_query = __ffs(ballot_result)-1;

	//Could not remove!
	if (thread_to_query == -1) return false;


	//int remove_index = __shfl_sync(0xfffffff, w, thread_to_query); 
	int remove_index = thread_to_query;



	//do the reverse move

	bool participating = (warpID > remove_index) && (warpID < get_fill());


	uint64_t val;

	if (participating){

		val = tags[warpID];
	}

	__syncwarp();

	if (participating){

		tags[warpID-1] = val;

	}

	//warp_utils::block_8_memmove_remove(warpID, tags, remove_index);


	__syncwarp();
	__threadfence();
	__syncwarp();

	return true;

}




__device__ bool megablock::assert_consistency(){

	return true;
	//return(popcnt(md[0]) == VIRTUAL_BUCKETS + 1);
}
