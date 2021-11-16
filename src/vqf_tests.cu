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


#include "include/vqf.cuh"

#include <openssl/rand.h>


#define BLOCK_SIZE 32

__global__ void test_insert_kernel(vqf* my_vqf, uint64_t * vals, uint64_t nvals){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	//if (tid > 0) return;
	if (tid >= nvals) return;

	my_vqf->insert(vals[tid]);

	printf("tid %llu done\n", tid);

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


	vqf * my_vqf =  build_vqf(1 << nbits);

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	test_insert_kernel<<<(nitems -1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(my_vqf, dev_vals, nitems);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nitems << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nitems/diff.count());



	free(vals);

	cudaFree(dev_vals);


	

	return 0;

}
