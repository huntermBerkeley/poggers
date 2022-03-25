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


#include "include/persistent_templated_vqf.cuh"
#include "include/metadata.cuh"

#include <openssl/rand.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void check_hits(bool * hits, uint64_t * misses, uint64_t nitems){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nitems) return;

	if (!hits[tid]){

		atomicAdd((unsigned long long int *) misses, 1ULL);

	}
}

template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> split_insert_timing(tcqf<Key, Val, Wrapper> * my_vqf, uint64_t * reference_vals, key_val_pair<Key, Val, Wrapper> * vals, uint64_t nvals, uint64_t * misses){




	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

	my_vqf->attach_lossy_buffers(reference_vals, vals, nvals);


	cudaDeviceSynchronize();
	
	gpuErrchk( cudaPeekAtLastError() );


	auto midpoint = std::chrono::high_resolution_clock::now();


	my_vqf->bulk_insert(misses);
	

	cudaDeviceSynchronize();

	gpuErrchk( cudaPeekAtLastError() );
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


	std::chrono::duration<double> attach_diff = midpoint-start;
  	std::chrono::duration<double> insert_diff = end-midpoint;	
  	std::chrono::duration<double> diff = end-start;



  	std::cout << "attached in " << attach_diff.count() << ", inserted in " << insert_diff.count() << ".\n";

  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> persistent_insert_timing(tcqf<Key, Val, Wrapper> * my_vqf, uint64_t * reference_vals, key_val_pair<Key, Val, Wrapper> * vals, uint64_t nvals, uint64_t * misses){


	key_val_pair<Key, Val, Wrapper> ** buffers;
	int * buffer_sizes;

	cudaMalloc((void **)& buffers, my_vqf->num_blocks*sizeof(key_val_pair<Key, Val, Wrapper> *));

	cudaMalloc((void **)& buffer_sizes, my_vqf->num_blocks*sizeof(int));

	//and boot
	

	gpuErrchk(cudaDeviceSynchronize());

	auto start = std::chrono::high_resolution_clock::now();

	my_vqf->prep_insert(nvals, reference_vals, vals, buffers, buffer_sizes);
	
	//full sync can happen
	gpuErrchk(cudaDeviceSynchronize());

	//sleep for a second to allow for boot up time
	//we can't synchronize while this is active
	//sleep(1);

	auto midpoint = std::chrono::high_resolution_clock::now();

	//my_vqf->bulk_insert(misses);
	

	
	my_vqf->boot_up();

	sleep(1);

	auto second_start = std::chrono::high_resolution_clock::now();

	my_vqf->submit_insert_only(nvals, reference_vals, vals, buffers, buffer_sizes);
	
	my_vqf->shut_down();


	gpuErrchk(cudaDeviceSynchronize());
	//cudaDeviceSynchronize();

	//gpuErrchk( cudaPeekAtLastError() );
	//and insert

	auto end = std::chrono::high_resolution_clock::now();

	//my_vqf->shut_down();

	
  	std::chrono::duration<double> attach_diff = midpoint-start;
  	std::chrono::duration<double> insert_diff = end - second_start;

  	std::chrono::duration<double> diff = end - second_start + midpoint-start;


   std::cout << "attached in " << attach_diff.count() << ", inserted in " << insert_diff.count() << ".\n";

  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/(diff.count()));

  	printf("Misses %llu\n", misses[0]);

  	gpuErrchk(cudaDeviceSynchronize());

  	misses[0] = 0;

  	gpuErrchk(cudaDeviceSynchronize());

  	gpuErrchk(cudaFree(buffers));
  	gpuErrchk(cudaFree(buffer_sizes));


  	

  	return diff;
}


template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ void bulk_query_timing(tcqf<Key, Val, Wrapper> * my_vqf, uint64_t * reference_vals, key_val_pair<Key, Val, Wrapper> * vals, uint64_t nvals, uint64_t * misses){



	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	
	my_vqf->attach_lossy_buffers(reference_vals, vals, nvals);
	my_vqf->bulk_query(hits);

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


template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ void fp_timing(tcqf<Key, Val, Wrapper> * my_vqf, uint64_t * reference_vals, key_val_pair<Key, Val, Wrapper> * vals, uint64_t nvals, uint64_t * misses){




	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	
	my_vqf->attach_lossy_buffers(reference_vals, vals, nvals);
	my_vqf->bulk_query(hits);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);



	//check hits

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "FP Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("FP Sorted Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu, ratio: %f\n", misses[0], 1.0 * (nvals - misses[0])/nvals);  

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


template <typename T>
__host__ T * generate_data(uint64_t nitems){


	//malloc space

	T * vals = (T *) malloc(nitems * sizeof(T));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



		to_fill += togen;

		printf("Generated %llu/%llu\n", to_fill, nitems);

	}

	return vals;
}

template <typename T>
__host__ T * load_main_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/main_data-32-data.txt";

	//char main_location[] = "/pscratch/sd/h/hunterm/vqf_data/main_data-32-data.txt";

	char * vals = (char * ) malloc(nitems * sizeof(T));

	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(T), pFile);

	if (result != nitems*sizeof(T)) abort();



	// //current supported format is no spacing one endl for the file terminator.
	// if (myfile.is_open()){


	// 	getline(myfile, line);

	// 	strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

	// 	myfile.close();
		

	// } else {

	// 	abort();
	// }


	return (T *) vals;


}

template <typename T>
__host__ T * load_alt_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/fp_data-32-data.txt";

	//char main_location[] = "/pscratch/sd/h/hunterm/vqf_data/fp_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(T));


	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(T), pFile);

	if (result != nitems*sizeof(T)) abort();



	return (T *) vals;


}

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);

	uint64_t num_batches = atoi(argv[2]);

	double batch_percent = 1.0 / num_batches;


	uint64_t nitems = (1ULL << nbits) * .85;


	//add one? just to guarantee that the clip is correct
	uint64_t items_per_batch = 1.05*nitems * batch_percent;


	printf("Starting test with %d bits, %llu items inserted in %d batches of %d.\n", nbits, nitems, num_batches, items_per_batch);


	using key_type = uint16_t;
	using main_data_type = key_val_pair<key_type>;

	uint64_t * val_references;
	uint64_t * dev_val_references;


	main_data_type * vals;
	main_data_type * dev_vals;


	val_references = load_main_data<uint64_t>(nitems);

	vals = load_main_data<main_data_type>(nitems);


	uint64_t * fp_val_references;

	main_data_type * fp_vals;

	//generate fp data to see comparison with true inserts
	fp_vals = load_alt_data<main_data_type>(nitems);

	fp_val_references = load_alt_data<uint64_t>(nitems);

	// vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	// RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);


	// other_vals = (uint64_t*) malloc(nitems*sizeof(other_vals[0]));

	// RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nitems);




	cudaMalloc((void ** )& dev_vals, items_per_batch*sizeof(main_data_type));

	cudaMalloc((void ** )& dev_val_references, items_per_batch*sizeof(uint64_t));

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
	
	//quad_hash_table * ht =  build_hash_table(1ULL << nbits);
	tcqf<key_type> * vqf = build_vqf<key_type>( (uint64_t)(1ULL << nbits));

	std::chrono::duration<double> diff = std::chrono::nanoseconds::zero();




	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();


	//dumb_testing
	// vqf->boot_up();

	// vqf->shut_down();

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

		cudaMemcpy(dev_val_references, val_references + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(main_data_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		//launch inserts
		diff += persistent_insert_timing<key_type>(vqf, dev_val_references, dev_vals, items_to_insert, misses);

		cudaDeviceSynchronize();

		cudaMemcpy(dev_val_references, val_references + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(main_data_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//launch queries
		//bulk_query_timing<key_type>(vqf, dev_val_references, dev_vals, items_to_insert, misses);


		cudaDeviceSynchronize();

		cudaMemcpy(dev_val_references, fp_val_references + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_vals, fp_vals + start, items_to_insert*sizeof(main_data_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//false queries
		//fp_timing<key_type>(vqf, dev_val_references, dev_vals, items_to_insert, misses);


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

	free_vqf(vqf);

	

	return 0;

}
