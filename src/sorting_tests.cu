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


#include "include/sorting_helper.cuh"
#include <openssl/rand.h>



__global__ void single_thread_power_of_two(){


	assert(greatest_power_of_two(17) == 16);
	assert(greatest_power_of_two(15) == 8);
	assert(greatest_power_of_two(10) == 8);
	assert(greatest_power_of_two(8) == 4);
	assert(greatest_power_of_two(7) == 4);
	assert(greatest_power_of_two(2) == 1);
	assert(greatest_power_of_two(3) == 2);

}

__global__ void warp_sort(uint64_t * items, int nitems){


	int warpID = (threadIdx.x + blockIdx.x*blockDim.x) % 32;


	bitonicSort(items, 0, nitems, true, warpID);

}

__global__ void byte_warp_sort(uint64_t * items, int nitems){


	int warpID = (threadIdx.x + blockIdx.x*blockDim.x) % 32;

	//short_warp_sort(items, nitems, 0, warpID);

	byteBitonicSort(items, 0, nitems, true, warpID);

}

__host__ bool assert_sorted(uint64_t * items, uint64_t nitems){


		if (nitems < 1) return true;

		uint64_t smallest = items[0];

		for (int i=1; i< nitems; i++){

		if (items[i] < smallest) return false;

		smallest = items[i];


		}

		return true;

}

__global__ void merge_dual_arrays_kernel(uint8_t * primary, uint8_t * secondary, int primary_nitems, int secondary_nitems){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	int teamID = (tid % blockDim.x) / 32;

	int warpID = tid % 32;

	merge_dual_arrays(primary, secondary, primary_nitems, secondary_nitems, teamID, warpID);

}


__global__ void sorting_network_kernel(uint8_t * items, int nitems){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	int warpID = tid % 32;

	sorting_network_8_bit(items, nitems, warpID);
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


__host__ uint8_t * generate_short_data(uint64_t nitems){


	//malloc space

	uint8_t * vals = (uint8_t *) malloc(nitems * sizeof(uint8_t));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(uint8_t));



		to_fill += togen;

		//printf("Generated %llu/%llu\n", to_fill, nitems);

	}

	return vals;
}


int main(int argc, char** argv) {


	//first check

	// single_thread_power_of_two<<<1,1>>>();


	// uint64_t * items;

	// cudaMallocManaged((void **)& items, sizeof(uint64_t)*100000);

	// cudaDeviceSynchronize();

	// items[0] = 1;

	// items[1] = 0;



	// warp_sort<<<32,1>>>(items, 2);



	// cudaDeviceSynchronize();

	// assert_sorted(items, 2);

	// cudaDeviceSynchronize();


	// items[0] = 3;
	// items[1] = 1;
	// items[2] = 0;
	// items[3] = 2;

	// cudaDeviceSynchronize();

	// warp_sort<<<32,1>>>(items, 4);



	// cudaDeviceSynchronize();

	// assert_sorted(items, 4);

	// cudaDeviceSynchronize();


	// uint64_t * data = generate_data(100000);

	// cudaMemcpy(items, data, sizeof(uint64_t)*100000, cudaMemcpyHostToDevice);

	// cudaDeviceSynchronize();

	// warp_sort<<<32,1>>>(items,100000);

	// cudaDeviceSynchronize();


	// assert_sorted(items, 100000);

	// cudaDeviceSynchronize();


	// cudaMemcpy(items, data, sizeof(uint64_t)*100000, cudaMemcpyHostToDevice);


	// cudaDeviceSynchronize();


	// byte_warp_sort<<<32,1>>>(items, 100000);

	// cudaDeviceSynchronize();

	// byte_assert_sorted(items, 100000);


	// cudaDeviceSynchronize();

	//tests for merge join

	uint8_t * primary;

	uint8_t * secondary;

	cudaMallocManaged((void **)& primary, sizeof(uint8_t)*10);

	cudaMallocManaged((void **)& secondary, sizeof(uint8_t)*4);

	primary[0] = 1;
	primary[1] = 2;
	primary[2] = 3;
	primary[3] = 10;

	secondary[0] = 0;
	secondary[1] = 4;
	secondary[2] = 12;
	secondary[3] = 15;

	cudaDeviceSynchronize();

	assert(short_byte_assert_sorted(primary, 4));

	assert(short_byte_assert_sorted(secondary, 4));

	cudaDeviceSynchronize();

	int nitems = 4;

	merge_dual_arrays_kernel<<<1,32>>>(primary, secondary, nitems, 4);



	cudaDeviceSynchronize();

	assert(short_byte_assert_sorted(primary, 8));

	cudaDeviceSynchronize();



	primary[0] = 1;
	primary[1] = 0;


	cudaDeviceSynchronize();

	sorting_network_kernel<<<1,32>>>(primary,2);

	cudaDeviceSynchronize();

	assert(short_byte_assert_sorted(primary, 2));

	cudaDeviceSynchronize();



	primary[0] = 5;
	primary[1] = 2;

	primary[2] = 1;

	primary[3] = 4;



	cudaDeviceSynchronize();

	sorting_network_kernel<<<1,32>>>(primary,4);

	cudaDeviceSynchronize();

	assert(short_byte_assert_sorted(primary, 4));

	cudaDeviceSynchronize();


	uint8_t * temp_items;

	cudaMallocManaged((void **)& temp_items, sizeof(uint8_t)*32);

	for (int i =0; i <200; i++){


		uint8_t * host_temp_items = generate_short_data(32);

		cudaMemcpy(temp_items, host_temp_items, sizeof(uint8_t)*32, cudaMemcpyHostToDevice);

		free(host_temp_items);

		cudaDeviceSynchronize();

		sorting_network_kernel<<<1,32>>>(temp_items, 32);

		cudaDeviceSynchronize();

		assert(short_byte_assert_sorted(temp_items, 32));

		cudaDeviceSynchronize();


	}
	



	printf("All tests succeeded\n");

	return 0;

}
