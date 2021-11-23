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


#define BLOCK_SIZE 1024

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



//reduce to see which bit 
__device__ unsigned int uint_pdep(uint64_t tid, unsigned int reduce, unsigned int deposit){


	int laneId = tid & 0x1f;

	int val = (reduce >> laneId) & 1;

	int final_val = val;

	//reduce val
	for (int i=1; i<=16; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, final_val, i, 32);
        if ((laneId) >= i)
            final_val += n;
    }

    final_val = final_val-val;
    //final_val should now contain the sume of the first i bits
    //prefix sum style

    //final_val & val should tell me if I want to write

    if (val){
    	printf("Thread %llu: %d %d\n", tid, final_val, val);
    }

    __syncwarp();

    unsigned int output_val = (val & (deposit >> laneId) & 1) << final_val;

    for (int offset = 16; offset > 0; offset /= 2)
    output_val += __shfl_down_sync(0xffffffff, output_val, offset);


	//output_val is now correct iff you are thread 0;

	return output_val;



}


__device__ uint64_t uint64_pdep(int laneId, uint64_t src, uint64_t mask){

	//int laneId = tid & 0x1f;

	///grab the two bits I am in charge of
	unsigned int val = (mask >> laneId*2) & 3;

	//for now popcount
	//since its only 4 value

	int pop_val = __popc(val);
	int final_val = pop_val;

	for (int i=1; i<=16; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, final_val, i, 32);
        if ((laneId) >= i)
            final_val += n;
    }

    final_val = final_val - pop_val;

    //printf("Thread %llu: %d %d\n", tid, final_val, pop_val);

    uint64_t output_val = (val & (src >> laneId*2) & 3) << final_val;

    for (int offset = 16; offset > 0; offset /= 2)
    output_val += __shfl_down_sync(0xffffffff, output_val, offset);

	return output_val;
}

//print a unsigned int as binary
__device__ void printUint(unsigned int val){


	for (int i=0; i<32; i++){

		printf("%d",(val >> i) & 1);
	}


}

//print a unsigned int as binary
__device__ void printUint64_t(uint64_t val){


	for (int i=0; i<64; i++){

		printf("%d", (unsigned int) ((val >> i) & 1));
	}


}

__device__ uint64_t warp_reduce(uint64_t tid){

	int laneId = tid & 0x1f;

	int value = laneId;

	for (int i=1; i<=16; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 32);
        if ((laneId) >= i)
            value += n;
    }

    return value;

}


__global__ void reduce_kernel(uint64_t nvals){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	if (tid > nvals) return;

	uint64_t val = warp_reduce(tid);

	printf("Thread %llu reporting %llu\n", tid, val);
}


__global__ void pdep_kernel(uint64_t nvals, unsigned int reduce, unsigned int deposit){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	if (tid >= nvals) return;

	unsigned int val = uint_pdep(tid, reduce, deposit);


	if (tid ==0){

		printf("reduce:  ");
		printUint(reduce);
		printf("\ndeposit: ");
		printUint(deposit);
		printf("\nresult:  ");
		printUint(val);
		printf("\n");
	}

}

__global__ void pdep_64_kernel(uint64_t nvals, uint64_t reduce, uint64_t deposit){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	if (tid >= nvals) return;

	unsigned int val = uint64_pdep(tid, reduce, deposit);


	if (tid ==0){

		printf("src : ");
		printUint64_t(reduce);
		printf("\nmask: ");
		printUint64_t(deposit);
		printf("\nres : ");
		printUint64_t(val);
		printf("\n");
	}

}



// //parallel bit deposit - given a mask and 
// __device__ void warp_pdep(int warpID, int maskID){


// 	__shared__ uint64_t 
// }

__global__ void pdep_timing_kernel(uint64_t nvals, uint64_t * reduce, uint64_t * deposit){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t globalwarpID = (tid/32);

	uint64_t threadID = tid%32;

	if (globalwarpID >= nvals) return;

	uint64_pdep(tid, reduce[globalwarpID], deposit[globalwarpID]);

}



__device__ int select(int warpID, uint64_t val, int bit){


	//slow version
	//I can do this with a precompute table
	//should save cycles?


	uint64_t mask1 = (2ULL << (warpID*2))-1;
	uint64_t mask2 = (2ULL << (warpID*2));

	int count1 = __popcll(mask1 & val)-1;

	//this can be faster
	int count2 = __popcll(mask2 & val)+count1;


	//printf("%d: %d\n%d: %d\n", 2*warpID, count1, 2*warpID+1, count2);

	int ballot = 0;

	int count_value = 0;

	if (count1 == bit || count2 == bit){
		ballot=1;

		count_value = warpID*2;

		if (count1 != bit){

			//addition is probably cheaper than another value set
			count_value += 1;

		}
	}

	//ballot bits together
	unsigned int ballot_result = __ballot_sync(0xffffffff, ballot);


	int thread_to_query = __ffs(ballot_result)-1;

	if (thread_to_query == -1) return -1;

	count_value = __shfl_sync(0xffffffff, count_value, thread_to_query); 

	return count_value;

}

__global__ void select_64_kernel(uint64_t nvals, uint64_t val, int bit){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	if (tid >= nvals) return;

	int i = bit;


	val = 137438953470ULL;
	
	unsigned int val_bit = select(tid, val, i);


	if (tid ==0){

		printf("\n %dth bit set at pos %d\n", i, val_bit);
		printf("val:  ");
		printUint64_t(val);
		printf("\nmod:  ");
		printUint64_t(1ULL << val_bit);
		printf("\n");
	}

	}

	


int main(int argc, char** argv) {


	uint64_t nbits = atoi(argv[1]);

	uint64_t bit = atoi(argv[2]);


	//select_64_kernel(uint64_t nvals, uint64_t reduce, uint64_t deposit)

	select_64_kernel<<<1, 32>>>(32, nbits, bit);

	cudaDeviceSynchronize();
	

	// uint64_t nbits = atoi(argv[1]);


	// uint64_t nitems = (1 << nbits) * .9;

	// uint64_t * vals;
	// uint64_t * dev_vals;

	// vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	// RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);

	// cudaMalloc((void ** )& dev_vals, nitems*sizeof(vals[0]));

	// cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);



	// uint64_t * deposits;

	// uint64_t * dev_deposits;

	// deposits = (uint64_t*) malloc(nitems*sizeof(deposits[0]));

	// RAND_bytes((unsigned char *)deposits, sizeof(*deposits) * nitems);


	// cudaMalloc((void ** )& dev_deposits, nitems*sizeof(deposits[0]));

	// cudaMemcpy(dev_deposits, deposits, nitems * sizeof(deposits[0]), cudaMemcpyHostToDevice);


	



	// cudaDeviceSynchronize();

	// printf("Setup wrapped up\n");
	// auto start = std::chrono::high_resolution_clock::now();



	// pdep_timing_kernel<<<(nitems*32 -1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(nitems, dev_vals, dev_deposits);

	// cudaDeviceSynchronize();


	// auto end = std::chrono::high_resolution_clock::now();


 //  	std::chrono::duration<double> diff = end-start;


 //  	std::cout << "Inserted " << nitems << " in " << diff.count() << " seconds\n";

 //  	printf("Inserts per second: %f\n", nitems/diff.count());

	

	return 0;

}
