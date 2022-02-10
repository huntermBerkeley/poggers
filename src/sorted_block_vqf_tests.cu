/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
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


#include "include/sorted_block_vqf.cuh"
#include "include/metadata.cuh"

#include <openssl/rand.h>






__host__ void insert_timing(optimized_vqf * my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	my_vqf->sorted_bulk_insert(vals, nvals, misses);
	

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


__global__ void test_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;




	if(!my_vqf->query(warpID, vals[teamID])){

		my_vqf->query(warpID, vals[teamID]);

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


__global__ void test_full_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

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


__host__ void full_query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_full_query_kernel<<<(32*nvals -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Full Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


__host__ void sort_timing(optimized_vqf * my_vqf){


	auto start = std::chrono::high_resolution_clock::now();


	my_vqf->sort_and_check();

	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Sorted in " << diff.count() << " seconds\n";


  	return;


}

__global__ void check_hits(bool * hits, uint64_t * misses, uint64_t nitems){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nitems) return;

	if (!hits[tid]){

		atomicAdd((unsigned long long int *) misses, 1ULL);

	}
}

__host__ void bulk_query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){



	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	

	my_vqf->bulk_query(vals, nvals, hits);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


__host__ void sorted_bulk_query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){



	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	

	my_vqf->sorted_bulk_query(vals, nvals, hits);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Sorted Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);  

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}



__host__ uint64_t * generate_data(uint64_t nitems){


	//malloc space

	uint64_t * vals = (uint64_t *) malloc(nitems * sizeof(uint64_t));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(uint64_t));



		to_fill += togen;

		printf("Generated %llu/%llu\n", to_fill, nitems);

	}

	return vals;
}

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);


	uint64_t nitems = (1ULL << nbits) * .9;

	uint64_t * vals;
	uint64_t * dev_vals;

	uint64_t * other_vals;
	uint64_t * dev_other_vals;


	vals = generate_data(nitems);

	// vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	// RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);


	// other_vals = (uint64_t*) malloc(nitems*sizeof(other_vals[0]));

	// RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nitems);




	cudaMalloc((void ** )& dev_vals, nitems*sizeof(vals[0]));

	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	bool * inserts;


	cudaMalloc((void ** )& inserts, nitems*sizeof(bool));

	cudaMemset(inserts, 0, nitems*sizeof(bool));



	// cudaMalloc((void ** )& dev_other_vals, nitems*sizeof(other_vals[0]));

	// cudaMemcpy(dev_other_vals, other_vals, nitems * sizeof(other_vals[0]), cudaMemcpyHostToDevice);


	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;


	optimized_vqf * my_vqf =  build_vqf(1ULL << nbits);


	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();

	


	cudaDeviceSynchronize();

	
	insert_timing(my_vqf, dev_vals, nitems,  misses);

	//cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	//sort_timing(my_vqf);

	cudaDeviceSynchronize();

	//query_timing(my_vqf, dev_vals, nitems,  misses);

    cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

// 	full_query_timing(my_vqf, dev_vals, nitems, misses);

	cudaDeviceSynchronize();


// 	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	sorted_bulk_query_timing(my_vqf, dev_vals, nitems, misses);

	cudaDeviceSynchronize();
	


// 	bulk_query_timing(my_vqf, dev_vals, nitems, misses);

// 	// cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

// 	// cudaDeviceSynchronize();


// 	//remove_timing(my_vqf, dev_vals, inserts, nitems,  misses);



// 	cudaDeviceSynchronize();

// //	sort_timing(my_vqf);

// 	cudaDeviceSynchronize();


// 	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

// 	cudaDeviceSynchronize();

// 	bulk_query_timing(my_vqf, dev_vals, nitems, misses);

// 	// cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

// 	// cudaDeviceSynchronize();


// 	//remove_timing(my_vqf, dev_vals, inserts, nitems,  misses);

// 	cudaDeviceSynchronize();
// 	//and insert

// 	//auto end = std::chrono::high_resolution_clock::now();


	free(vals);

	cudaFree(dev_vals);

	cudaFree(misses);

	

	return 0;

}
