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


#include "include/team_vqf.cuh"

#include <openssl/rand.h>


#define BLOCK_SIZE 512

__global__ void test_insert_kernel(vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;

	if (my_vqf->insert(warpID, vals[teamID])){


		assert(my_vqf->query(warpID, vals[teamID]));

		assert(my_vqf->remove(warpID, vals[teamID]));
	} else {


		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, 1);
	}



	//printf("tid %llu done\n", tid);

	// //does a single thread have this issue?
	// for (uint64_t i =0; i< nvals; i++){

	// 	assert(vals[i] != 0);

	// 	my_vqf->insert(vals[i]);

	// }

	
}


int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);


	uint64_t nitems = (1 << nbits) * .9;

	uint64_t * vals;
	uint64_t * dev_vals;

	vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);



	cudaMalloc((void ** )& dev_vals, nitems*sizeof(vals[0]));

	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;


	vqf * my_vqf =  build_vqf(1 << nbits);

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	test_insert_kernel<<<(32*nitems -1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(my_vqf, dev_vals, nitems, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nitems << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nitems/diff.count());

  	printf("Misses %llu\n", misses[0]);

	free(vals);

	cudaFree(dev_vals);

	cudaFree(misses);

	

	return 0;

}
