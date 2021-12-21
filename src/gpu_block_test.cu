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

#include "include/gpu_block.cuh"



// insert and remove an item multiple times
// assumes block already initialized
__device__ void insert_remove_test(gpu_block * block, uint64_t tid){


	if (tid >=32) return;

	int laneId = tid & 0x1f;

	block[0].lock(laneId);

	for (uint64_t i =0; i < 10; i++){

		block[0].insert(laneId, i);

		// if (laneId == 0){
		// 	printf("Fill: %d\n", block[0].get_fill());
		// }

		// __syncwarp();

		assert(block[0].get_fill() == i+1);

	}


	//now that items have been inserted, remove in order

	for (uint64_t i=0; i < 10; i++){

		assert(block[0].remove(laneId, i));

		assert(block[0].get_fill() == 9-i);
	}



	for (uint64_t i =0; i < 10; i++){

		block[0].insert(laneId, i);

		assert(block[0].get_fill() == i+1);

	}

	for (uint64_t i=0; i < 10; i++){

		assert(block[0].remove(laneId, 9-i));

		assert(block[0].get_fill() == 9-i);
	}

	for (uint64_t i =0; i < 10; i++){

		block[0].insert(laneId, 9-i);

		assert(block[0].get_fill() == i+1);

	}

	for (uint64_t i=0; i < 10; i++){

		assert(block[0].remove(laneId, 9-i));

		assert(block[0].get_fill() == 9-i);
	}


	block[0].unlock(laneId);


}

//insert multiple copies of the same item, and make sure the filter holds them correctly
// __device__ void insert_duplicates_test(vqf_block * block){


// 	for (uint64_t j =0; j < 1000; j++){


// 		for (uint64_t i =0; i < 20; i++){

// 			block[0].insert(j);
// 			block[0].remove(j);

// 			assert(block[0].get_fill() == 0);

// 			}


// 		for (uint64_t i =0; i < 20; i++){

// 			block[0].insert(j);

// 		}

// 		for (uint64_t i =0; i < 20; i++){

// 			block[0].remove(j);

			
// 		}

// 		assert(block[0].get_fill() == 0);


// 	}


// }

//test filling to a certain ration and unfilling
__device__ void insert_duplicates_test(int warpID, gpu_block * block){


	block[0].lock(warpID);


	for (int max_fill = 0; max_fill < 28; max_fill++){


		for (int i = 0; i < max_fill; i++){

			block[0].insert(warpID, i*50);

			if (block[0].get_fill() != i+1){

				if (warpID == 0){

					printf("Pre lock, failed at fill ratio %d, insert %d. Filled to %d, expected %d\n", max_fill, i, block[0].get_fill(), i+1);
				}
			}
			assert(block[0].get_fill() == i+1);
		}
		

		for (int i = 0; i < max_fill; i++){
			assert(block[0].query(warpID, i*50));
		}


		for (int i = 0; i < max_fill; i++){
			assert(block[0].remove(warpID, i*50));
		}

		//if (warpID == 0) printf("Fill: %llu\n", block[0].get_fill());
		assert(block[0].get_fill() == 0);

		if (warpID == 0) printf("Done with %d\n", max_fill);

	}


	block[0].unlock(warpID);



	if (warpID == 0) printf("Done with unlocked\n");



	block[0].lock(warpID);

	for (int max_fill = 0; max_fill < 28; max_fill++){


		for (int i = 0; i < max_fill; i++){
			block[0].insert(warpID, i*50);
			assert(block[0].assert_consistency());

			if (block[0].get_fill() != i+1){

				if (warpID == 0){
					printf("Post lock, failed at fill ratio %d, insert %d\n", warpID, i);
				}
			}

			assert(block[0].get_fill() == i+1);
		}
		

		for (int i = 0; i < max_fill; i++){
			block[0].query(warpID, i*50);
			assert(block[0].assert_consistency());
		}


		for (int i = 0; i < max_fill; i++){
			block[0].remove(warpID, i*50);
			assert(block[0].assert_consistency());
		}

		assert(block[0].get_fill() == 0);
		if (warpID == 0) printf("Done with %d\n", max_fill);


	}

	block[0].unlock(warpID);

}


//fill the block with junk items, then test queries and removes
// __device__ void junk_tests(vqf_block * block){


// 	for(uint64_t j = 25; j < 35; j++){
// 		block[0].insert(j);
// 	}


// 	for (uint64_t i=0; i< 10; i++){

// 		block[0].insert(i);

// 	}

// 	for (uint64_t i = 0; i < 10; i++)
// 	{
// 		assert(block[0].query(i));
// 	}

// 	//assert we can still find the original inserts
// 	for(uint64_t j = 25; j < 35; j++){
// 		assert(block[0].query(j));
// 	}


// 	for(uint64_t j = 25; j < 35; j++){
// 		block[0].remove(j);
// 	}

// 	for (uint64_t i = 0; i < 10; i++)
// 	{
// 		assert(block[0].query(i));
// 	}


// 	for(uint64_t i = 0; i < 10; i++){
// 		block[0].remove(i);
// 	}

// 	assert(block[0].get_fill()==0);

// }


// __device__ void lock_tests(vqf_block * block, uint64_t tid){


// 	block[0].lock();

// 	block[0].insert(tid);

// 	block[0].unlock();


// 	block[0].lock();

// 	assert(block[0].query(tid));

// 	block[0].remove(tid);

// 	assert(!block[0].query(tid));

// 	block[0].unlock();
// }

__global__ void attempt_lock(gpu_block * block, uint64_t nthreads){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nthreads) return;

	uint64_t warpID = tid / 32;

	int laneId = tid % 32;

	if (laneId == 0){
		block[warpID].setup();
	}
	


	for (int i =0; i < 10; i++){


	block[warpID].lock(laneId);

	//insert_remove_test(block);

	// printf("Passed insert/remove test\n");

	// insert_duplicates_test(block);

	// printf("Passed duplicate inserts\n");

	// junk_tests(block);

	// printf("Passed junk tests\n");

	block[warpID].insert(laneId, i*10);



	block[warpID].unlock(laneId);


	}

	for (int i =0; i < 10; i++){


	block[warpID].lock(laneId);

	//insert_remove_test(block);

	// printf("Passed insert/remove test\n");

	// insert_duplicates_test(block);

	// printf("Passed duplicate inserts\n");

	// junk_tests(block);

	// printf("Passed junk tests\n");

	assert(block[warpID].query(laneId, i*10));



	block[warpID].unlock(laneId);


	}


	for (int i =0; i < 10; i++){


	block[warpID].lock(laneId);

	//insert_remove_test(block);

	// printf("Passed insert/remove test\n");

	// insert_duplicates_test(block);

	// printf("Passed duplicate inserts\n");

	// junk_tests(block);

	// printf("Passed junk tests\n");

	assert(block[warpID].remove(laneId, i*10));



	block[warpID].unlock(laneId);


	}


	for (int i =0; i < 10; i++){


		block[warpID].lock(laneId);

		//insert_remove_test(block);

		// printf("Passed insert/remove test\n");

		// insert_duplicates_test(block);

		// printf("Passed duplicate inserts\n");

		// junk_tests(block);

		// printf("Passed junk tests\n");

		assert(!block[warpID].query(laneId, i*10));



		block[warpID].unlock(laneId);


	}


	for (int i =0; i < 10; i++){


		block[warpID].lock(laneId);

		//insert_remove_test(block);

		// printf("Passed insert/remove test\n");

		// insert_duplicates_test(block);

		// printf("Passed duplicate inserts\n");

		// junk_tests(block);

		// printf("Passed junk tests\n");

		assert(!block[warpID].remove(laneId, i*10));



		block[warpID].unlock(laneId);


	}

	insert_remove_test(block, tid);




	insert_duplicates_test(laneId, block);



}


// __global__ void lock_test_kernel(vqf_block * block, uint64_t nthreads){

// 	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

// 	if (tid >= nthreads) return;


// 	lock_tests(block, tid);

// }









int main(int argc, char** argv) {
	

	gpu_block * block;


	cudaMalloc((void **)&block, sizeof(gpu_block));

	attempt_lock<<<1, 32>>>(block, 10000);

	//lock_test_kernel<<<1,32>>>(block, 32);

	cudaDeviceSynchronize();

	cudaFree(block);

	

	return 0;

}
