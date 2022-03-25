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

#include <unistd.h>

#include "include/cuda_queue.cuh"

#include <openssl/rand.h>



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


__global__ void submit_task_to_queue(int tasknum, int task_type, cuda_queue<uint64_t> * queue){

	uint64_t tid = threadIdx.x+blockDim.x*blockIdx.x;


	if (tid != 0) return;

	submission_block<uint64_t> block;

	block.submissionID = tasknum;
	block.submission_type = task_type;

	queue->submit_task(&block);

	

}

__global__ void submit_tasks_and_wait(cuda_queue<uint64_t> * queue){

		uint64_t tid = threadIdx.x+blockDim.x*blockIdx.x;


		if (tid != 0) return;

		int cap = 10000;

		for (int i = 1; i < cap; i++){

			submission_block<uint64_t> block;

			block.submissionID = i;
			block.submission_type = 2;

			queue->submit_task(&block);

			//while(!queue->task_done(i));

			while (true){

				//printf("Stalling on wait %d\n", i);
				if (queue->task_done(i)){

					//printf("Wait broken!\n");
					break;


				} 
			}

		}

		printf("Done with submissions\n");

		submission_block<uint64_t> block;

		block.submissionID = cap;

		block.submission_type = 0;

		queue->submit_task(&block);

		return;



}

int main(int argc, char** argv) {

	printf("Booting.\n");
	fflush(stdout);

	cudaStream_t persistent_stream;

	cudaStreamCreate(&persistent_stream);

	cudaStream_t submit_stream;

	cudaStreamCreate(&submit_stream);
	

	// cuda_queue * queue = build_queue(40);

	// cudaDeviceSynchronize();

	// persistent_kernel<<<1,1,0,persistent_stream>>>(queue);


	// for (int i = 0; i < 10; i++){

	// 	submit_task_to_queue<<<1,1,0,submit_stream>>>(i+1, 1, queue);
	// }

	// //sleep(4);

	// for (int i = 10; i < 20; i++){

	// 	submit_task_to_queue<<<1,1,0,submit_stream>>>(i+1, 2, queue);

	// }

	// //sleep(4);


	// for (int i = 20; i < 30; i++){

	// 	submit_task_to_queue<<<1,1,0,submit_stream>>>(i+1, 1, queue);
	// }

	// //sleep(4);

	// submit_task_to_queue<<<1,1,0,submit_stream>>>(31, 0, queue);


	// cudaDeviceSynchronize();

	// free_queue(queue);


	cuda_queue<uint64_t> * queue = build_queue<uint64_t>((uint64_t) 5);


	cudaDeviceSynchronize();

	printf("Queue built\n");
	fflush(stdout);


	persistent_kernel_test<uint64_t><<<1,1,0,persistent_stream>>>(queue);

	printf("Main kernel launched.\n");
	fflush(stdout);



	auto start = std::chrono::high_resolution_clock::now();

	submit_tasks_and_wait<<<1,1,0,submit_stream>>>(queue);


	cudaStreamSynchronize(submit_stream);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end-start;


  	std::cout << "Took "  << diff.count() << " seconds\n";



	printf("secondary_kernel_launched.\n");
	fflush(stdout);

	//sleep(4);
	fflush(stdout);

	cudaDeviceSynchronize();


	fflush(stdout);
	cudaDeviceSynchronize();

	cudaStreamDestroy(persistent_stream);
	cudaStreamDestroy(submit_stream);

	cudaDeviceSynchronize();

	printf("All done\n");
	fflush(stdout);


	return 0;

}
