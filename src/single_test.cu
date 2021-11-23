/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

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

#include "include/vqf_block.cuh"



//insert and remove an item multiple times
//assumes block already initialized
__device__ void insert_remove_test(vqf_block * block){

	for (uint64_t i =0; i < 10; i++){

		block[0].insert(i);

		assert(block[0].get_fill() == i+1);

	}


	//now that items have been inserted, remove in order

	for (uint64_t i=0; i < 10; i++){

		assert(block[0].remove(i));

		assert(block[0].get_fill() == 9-i);
	}



	for (uint64_t i =0; i < 10; i++){

		block[0].insert(i);

		assert(block[0].get_fill() == i+1);

	}

	for (uint64_t i=0; i < 10; i++){

		assert(block[0].remove(9-i));

		assert(block[0].get_fill() == 9-i);
	}

	for (uint64_t i =0; i < 10; i++){

		block[0].insert(9-i);

		assert(block[0].get_fill() == i+1);

	}

	for (uint64_t i=0; i < 10; i++){

		assert(block[0].remove(9-i));

		assert(block[0].get_fill() == 9-i);
	}


}

//insert multiple copies of the same item, and make sure the filter holds them correctly
__device__ void insert_duplicates_test(vqf_block * block){


	for (uint64_t j =0; j < 1000; j++){


		for (uint64_t i =0; i < 20; i++){

			block[0].insert(j);
			block[0].remove(j);

			assert(block[0].get_fill() == 0);

			}


		for (uint64_t i =0; i < 20; i++){

			block[0].insert(j);

		}

		for (uint64_t i =0; i < 20; i++){

			block[0].remove(j);

			
		}

		assert(block[0].get_fill() == 0);


	}


}


//fill the block with junk items, then test queries and removes
__device__ void junk_tests(vqf_block * block){


	for(uint64_t j = 25; j < 35; j++){
		block[0].insert(j);
	}


	for (uint64_t i=0; i< 10; i++){

		block[0].insert(i);

	}

	for (uint64_t i = 0; i < 10; i++)
	{
		assert(block[0].query(i));
	}

	//assert we can still find the original inserts
	for(uint64_t j = 25; j < 35; j++){
		assert(block[0].query(j));
	}


	for(uint64_t j = 25; j < 35; j++){
		block[0].remove(j);
	}

	for (uint64_t i = 0; i < 10; i++)
	{
		assert(block[0].query(i));
	}


	for(uint64_t i = 0; i < 10; i++){
		block[0].remove(i);
	}

	assert(block[0].get_fill()==0);

}


__device__ void lock_tests(vqf_block * block, uint64_t tid){


	block[0].lock();

	block[0].insert(tid);

	block[0].unlock();


	block[0].lock();

	assert(block[0].query(tid));

	block[0].remove(tid);

	assert(!block[0].query(tid));

	block[0].unlock();
}

__global__ void attempt_lock(vqf_block * block, uint64_t nthreads){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nthreads) return;


	block[0].setup();

	block[0].lock();

	insert_remove_test(block);

	printf("Passed insert/remove test\n");

	insert_duplicates_test(block);

	printf("Passed duplicate inserts\n");

	junk_tests(block);

	printf("Passed junk tests\n");


	block[0].unlock();


}


__global__ void lock_test_kernel(vqf_block * block, uint64_t nthreads){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nthreads) return;


	lock_tests(block, tid);

}









int main(int argc, char** argv) {
	

	vqf_block * block;


	cudaMalloc((void **)&block, sizeof(vqf_block));

	attempt_lock<<<1, 1>>>(block, 10000);

	lock_test_kernel<<<1,32>>>(block, 32);

	cudaDeviceSynchronize();

	cudaFree(block);

	

	return 0;

}
