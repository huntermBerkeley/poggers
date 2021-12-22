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

#include "include/optimized_vqf.cuh"
#include "include/metadata.cuh"

//QUERY CODE

 __global__ void test_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;


	if(!my_vqf->full_query(warpID, vals[teamID])){

		my_vqf->full_query(warpID, vals[teamID]);

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


__host__ void query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_query_kernel<<<(32*nvals -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}




int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);


	uint64_t nitems = (1ULL << nbits) * .8;

	uint64_t * vals;
	uint64_t * dev_vals;

	vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);


	// other_vals = (uint64_t*) malloc(nitems*sizeof(other_vals[0]));

	// RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nitems);




	cudaMalloc((void ** )& dev_vals, nitems*sizeof(vals[0]));

	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;


	optimized_vqf * my_vqf =  build_vqf(1 << nbits);


	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();




	my_vqf->insert_power_of_two(dev_vals, nitems);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nitems << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nitems/diff.count());


  	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	query_timing(my_vqf, dev_vals, nitems, misses);


  	//setup for queries

	free(vals);

	cudaFree(dev_vals);

	cudaFree(misses);

	

	return 0;

}
