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


#include "include/gpu_quad_hash_table.cuh"
#include "include/hash_metadata.cuh"

#include <openssl/rand.h>




__global__ void bulk_insert_kernel(quad_hash_table * ht, uint64_t * vals, uint64_t nvals, uint64_t * misses){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nvals) return;


	if (!ht->insert(vals[tid])) atomicAdd( (unsigned long long int *) misses, 1);

	return;

}

__host__ std::chrono::duration<double> insert_timing(quad_hash_table * ht, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	bulk_insert_kernel<<<(nvals -1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(ht, vals, nvals, misses);
	

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


  	return diff;



}




__global__ void bulk_query_kernel(quad_hash_table * ht, uint64_t * vals, uint64_t nvals, uint64_t * misses){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nvals) return;


	if (!ht->query(vals[tid])) atomicAdd( (unsigned long long int *) misses, 1);

	return;

}



__host__ void query_timing(quad_hash_table * ht, uint64_t * vals, uint64_t nvals, uint64_t * misses){


	cudaDeviceSynchronize();


	auto start = std::chrono::high_resolution_clock::now();


	bulk_query_kernel<<<(nvals -1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(ht, vals, nvals, misses);
	


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


__host__ void fp_timing(quad_hash_table * ht, uint64_t * vals, uint64_t nvals, uint64_t * misses){



	

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	
	bulk_query_kernel<<<(nvals -1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(ht, vals, nvals, misses);
	

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "FP Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("FP Sorted Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu, ratio: %f\n", misses[0], 1.0 * (nvals - misses[0])/nvals);  

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


__host__ uint64_t * load_main_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/main_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(uint64_t));

	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(uint64_t), pFile);

	if (result != nitems*sizeof(uint64_t)) abort();



	// //current supported format is no spacing one endl for the file terminator.
	// if (myfile.is_open()){


	// 	getline(myfile, line);

	// 	strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

	// 	myfile.close();
		

	// } else {

	// 	abort();
	// }


	return (uint64_t *) vals;


}

__host__ uint64_t * load_alt_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/fp_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(uint64_t));


	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(uint64_t), pFile);

	if (result != nitems*sizeof(uint64_t)) abort();



	return (uint64_t *) vals;


}

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);

	uint64_t num_batches = atoi(argv[2]);

	double batch_percent = 1.0 / num_batches;


	uint64_t nitems = (1ULL << nbits) * .85;


	//add one? just to guarantee that the clip is correct
	uint64_t items_per_batch = 1.05*nitems * batch_percent;


	printf("Starting test with %d bits, %llu items inserted in %d batches of %d.\n", nbits, nitems, num_batches, items_per_batch);




	uint64_t * vals;
	uint64_t * dev_vals;

	uint64_t * other_vals;
	uint64_t * dev_other_vals;


	vals = load_main_data(nitems);


	uint64_t * fp_vals;

	//generate fp data to see comparison with true inserts
	fp_vals = load_alt_data(nitems);

	// vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	// RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);


	// other_vals = (uint64_t*) malloc(nitems*sizeof(other_vals[0]));

	// RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nitems);




	cudaMalloc((void ** )& dev_vals, items_per_batch*sizeof(vals[0]));

	//cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	//bool * inserts;


	// cudaMalloc((void ** )& inserts, items_per_batch*sizeof(bool));

	// cudaMemset(inserts, 0, items_per_batch*sizeof(bool));



	// cudaMalloc((void ** )& dev_other_vals, nitems*sizeof(other_vals[0]));

	// cudaMemcpy(dev_other_vals, other_vals, nitems * sizeof(other_vals[0]), cudaMemcpyHostToDevice);


	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;


	//change the way vqf is built to better suit test and use cases? TODO with active reconstruction for exact values / struct support
	quad_hash_table * ht =  build_hash_table(1ULL << nbits);


	std::chrono::duration<double> diff = std::chrono::nanoseconds::zero();




	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();

	

	for (int batch = 0; batch< num_batches; batch++){

		//calculate size of segment

		printf("Batch %d:\n", batch);

		//runs from batch/num_batches*nitems to batch
		uint64_t start = batch*nitems/num_batches;
		uint64_t end = (batch+1)*nitems/num_batches;
		if (end > nitems) end = nitems;

		uint64_t items_to_insert = end-start;


		assert(items_to_insert < items_per_batch);

		//prep dev_vals for this round
		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(vals[0]), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		//launch inserts
		diff += insert_timing(ht, dev_vals, items_to_insert, misses);

		cudaDeviceSynchronize();

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(vals[0]), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//launch queries
		query_timing(ht, dev_vals, items_to_insert, misses);


		cudaDeviceSynchronize();

		cudaMemcpy(dev_vals, fp_vals + start, items_to_insert*sizeof(vals[0]), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//false queries
		fp_timing(ht, dev_vals, items_to_insert, misses);


		cudaDeviceSynchronize();


		//keep some organized spacing
		printf("\n\n");

		fflush(stdout);

		cudaDeviceSynchronize();



	}

	printf("Tests Finished.\n");

	std::cout << "Queried " << nitems << " in " << diff.count() << " seconds\n";

	printf("Final speed: %f\n", nitems/diff.count());

	free(vals);

	free(fp_vals);

	cudaFree(dev_vals);

	cudaFree(misses);

	

	return 0;

}
