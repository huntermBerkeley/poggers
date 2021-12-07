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
#include "include/warp_utils.cuh"

namespace warp_utils {



//reduce to see which bit 
__device__ unsigned int uint_pext(int laneId, unsigned int reduce, unsigned int deposit){


	//int laneId = tid & 0x1f;

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

    unsigned int output_val = (val & (deposit >> laneId) & 1) << final_val;

    for (int offset = 16; offset > 0; offset /= 2)
    output_val += __shfl_down_sync(0xffffffff, output_val, offset);


	//output_val is now correct iff you are thread 0;

	return output_val;



}


__device__ uint64_t uint64_pext(int laneId, uint64_t reduce, uint64_t deposit){

	//int laneId = tid & 0x1f;

	///grab the two bits I am in charge of
	unsigned int val = (reduce >> laneId*2) & 3;

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

    uint64_t output_val = (val & (deposit >> laneId*2) & 3) << final_val;

    for (int offset = 16; offset > 0; offset /= 2)
    output_val += __shfl_down_sync(0xffffffff, output_val, offset);

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

// __device__ uint64_t warp_reduce(int laneId){

// 	int laneId = tid & 0x1f;

// 	int value = laneId;

// 	for (int i=1; i<=16; i*=2) {
//         // We do the __shfl_sync unconditionally so that we
//         // can read even from threads which won't do a
//         // sum, and then conditionally assign the result.
//         int n = __shfl_up_sync(0xffffffff, value, i, 32);
//         if ((laneId) >= i)
//             value += n;
//     }

//     return value;

// }



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


//move up to 32 bytes at a time using the warp, each thread takes a different byte at a time
__device__ void warp_memmove(int warpID, void* dst, const void* src, size_t n){

	char * char_dst = (char *) dst;
	char * char_src = (char *) src;

  //double check this,
  //think it is just > since dst+n does not get copied

  

  if (char_src+n > char_dst && char_src < char_dst){

  	//copy backwards 
  	for (int i =n-1; i >= 0; i-=32){

  		int my_index = i - warpID;

  		char my_value;

  		if (my_index >= 0){

  			my_value = char_src[my_index];
  			

  		}


  		__syncwarp();

  		if (my_index >= 0){

  			char_dst[my_index] = my_value;
  		}

  		__syncwarp();
  		

  	}

  } else {

  	//copy regular
  	for (int i =0; i<n; i+=32){


  		int my_index = i+warpID;

  		char my_value;

  		if (my_index < n){
  			my_value = char_src[my_index];
  		}

  		__syncwarp();

  		if (my_index < n){
  			char_dst[my_index] = my_value;
  		}

  		__syncwarp();
  	}


  }


  __threadfence();

}


//shuffle items to the right
//this code is as specialised as possible
//to reduce the sass code generated for it
//warp_memmove is 294 lines with ~50% sync
//only 28 tags, so if you're below we don't care
__device__ void block_8_memmove_insert(int warpID, uint16_t * tags, uint16_t tag, int index){

    uint16_t old;

	bool participating = (warpID >= index && warpID < 27);

	if (participating){


		//gather indices
		old = tags[warpID];

	}
	__syncwarp();


	if (participating){
		tags[warpID+1] = old;
		
	}

	if (warpID == 0){
		tags[index] = tag;
	}

	__syncwarp();

	return;

}


//shuffle items to the right
//this code is as specialised as possible
//to reduce the sass code generated for it
//warp_memmove is 294 lines with ~50% sync
//only 28 tags, so if you're below we don't care
__device__ void block_8_memmove_remove(int warpID, uint16_t * tags, int index){

    uint16_t old;

	bool participating = warpID > index && warpID < 28;

	if (participating){


		//gather indices
		old = tags[warpID];

	}
	__syncwarp();


	if (participating){
		tags[warpID-1] = old;
	}

	__syncwarp();

	return;

}





}
