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



#include <openssl/rand.h>


#define BLOCK_SIZE 256

__global__ void test_insert_kernel(uint64_t nvals){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	__shared__ int test_mem[BLOCK_SIZE];

	//if (tid > 0) return;
	if (tid >= nvals) return;

	uint64_t warpID = (tid/32)  % (BLOCK_SIZE/32);;

	uint64_t threadID = tid%32;

	test_mem[warpID*32+threadID] = 1;
	

	__syncwarp();



	assert(test_mem[warpID*32 + ((threadID +1) % 32)] == 1);


}


__device__ void warp_test(uint64_t warpID, uint64_t threadID){


	__shared__ int dev_test_mem[BLOCK_SIZE];

	dev_test_mem[warpID*32+threadID] = 1;


	__syncwarp();

	assert(dev_test_mem[warpID*32 + ((threadID+1)%32)] == 1);


}

__global__ void dev_insert_kernel(uint64_t nvals){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid > nvals) return;

	uint64_t warpID = (tid/32) % (BLOCK_SIZE/32);

	uint64_t threadID = tid % 32;

	warp_test(warpID, threadID);
}



// //parallel bit deposit - given a mask and 
// __device__ void warp_pdep(int warpID, int maskID){


// 	__shared__ uint64_t 
// }

// __device__ uint64_t warp_reduce(uint64_t tid){

// 	int laneId = tid & 0x1f;

// 	for (int i=1; i<=16; i*=2) {
//         // We do the __shfl_sync unconditionally so that we
//         // can read even from threads which won't do a
//         // sum, and then conditionally assign the result.
//         int n = __shfl_up_sync(0xffffffff, value, i, 32);
//         if ((laneId) >= i)
//             value += n;
//     }
// }

int main(int argc, char** argv) {
	




	uint64_t nitems = 1024;


	cudaDeviceSynchronize();

	test_insert_kernel<<<(nitems -1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(nitems);

	cudaDeviceSynchronize();

	dev_insert_kernel<<<(nitems -1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(nitems);

	cudaDeviceSynchronize();




	

	return 0;

}
