
#ifndef SORTING_HELPER_C
#define SORTING_HELPER_C


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/sorting_helper.cuh"
#include "include/metadata.cuh"

#include <assert.h>



__device__ void swap(uint64_t * items, int i, int j){

	uint64_t temp = items[i];

	items[i] = items[j];
	items[j] = temp;

}

__device__ int greatest_power_of_two(int n){

	int k=1;

	while (k>0 && k<n) k = k << 1;

	return k >> 1;
}


__device__ void compare_and_swap(uint64_t * items, int i, int j, bool dir){


	


		if (dir ==(items[i] > items[j])) swap(items, i, j);

}

__device__ void bitonicMerge(uint64_t * items, int low, int count, bool dir, int warpID){

	if (count > 1){

		int k = greatest_power_of_two(count);

		for (int i = low +warpID; i < low + count - k ; i+=32){

			compare_and_swap(items, i, i+k, dir);
		}
		__syncwarp();


		bitonicMerge(items, low, k, dir, warpID);
		bitonicMerge(items, low+k, count-k, dir, warpID);

	}

}


__device__ void bitonicSort(uint64_t * items, int low, int count, bool dir, int warpID){


	if (count > 1){

		int k = count/2;

		//sort the start in ascending order
		bitonicSort(items, low, k, !dir, warpID);


		//sort the lower half in descending order
		bitonicSort(items, low+k, count-k, dir, warpID);


		//merge the sequence in ascending order
		bitonicMerge(items, low, count, dir, warpID);

		__syncwarp();

	}

}



__device__ void byte_compare_and_swap(uint64_t * items, int i, int j, bool dir){


	


		if (dir ==( (items[i] & 0xFF) > (items[j] & 0xFF) )) swap(items, i, j);

}

__device__ void byteBitonicMerge(uint64_t * items, int low, int count, bool dir, int warpID){

	if (count > 1){

		int k = greatest_power_of_two(count);

		for (int i = low +warpID; i < low + count - k ; i+=32){

			byte_compare_and_swap(items, i, i+k, dir);
		}
		__syncwarp();


		byteBitonicMerge(items, low, k, dir, warpID);
		byteBitonicMerge(items, low+k, count-k, dir, warpID);

	}

}


__device__ void byteBitonicSort(uint64_t * items, int low, int count, bool dir, int warpID){


	if (count > 1){

		int k = count/2;

		//sort the start in ascending order
		byteBitonicSort(items, low, k, !dir, warpID);


		//sort the lower half in descending order
		byteBitonicSort(items, low+k, count-k, dir, warpID);


		//merge the sequence in ascending order
		byteBitonicMerge(items, low, count, dir, warpID);

		__syncwarp();


	}


}

__device__ void byteSwap(uint8_t * items, int i, int j){

	uint64_t temp = items[i];

	items[i] = items[j];
	items[j] = temp;

}

__device__ void short_byte_compare_and_swap(uint8_t * items, int i, int j, bool dir){



		if (dir ==( (items[i]) > (items[j]) )) byteSwap(items, i, j);

}

__device__ void shortByteBitonicMerge(uint8_t * items, int low, int count, bool dir, int warpID){

	if (count > 1){

		int k = greatest_power_of_two(count);

		for (int i = low +warpID; i < low + count - k ; i+=32){

			short_byte_compare_and_swap(items, i, i+k, dir);
		}
		__syncwarp();


		shortByteBitonicMerge(items, low, k, dir, warpID);
		shortByteBitonicMerge(items, low+k, count-k, dir, warpID);

	}

}

__device__ void shortByteBitonicSort(uint8_t * items, int low, int count, bool dir, int warpID){


		if (count > 1){

		int k = count/2;

		//sort the start in ascending order
		shortByteBitonicSort(items, low, k, !dir, warpID);


		//sort the lower half in descending order
		shortByteBitonicSort(items, low+k, count-k, dir, warpID);


		//merge the sequence in ascending order
		shortByteBitonicMerge(items, low, count, dir, warpID);

		__syncwarp();


	}


}


__host__ __device__ bool byte_assert_sorted(uint64_t * items, uint64_t nitems){


		if (nitems < 1) return true;

		uint64_t smallest = items[0];

		for (int i=1; i< nitems; i++){

		if ( (items[i] & 0xFF) < ( smallest & 0xFF) ) return false;

		smallest = items[i];


		}

		return true;

}


__host__ __device__ bool two_byte_assert_sorted(uint64_t * items, uint64_t nitems){


		if (nitems < 1) return true;

		uint64_t smallest = items[0];

		for (int i=1; i< nitems; i++){

		if ( (items[i] & 0xFFFF) < ( smallest & 0xFFFF) ) return false;

		smallest = items[i];


		}

		return true;

}

__host__ __device__ bool short_byte_assert_sorted(uint8_t * items, uint64_t nitems){


	if (nitems < 1) return true;

	uint8_t smallest = items[0];

	for (int i=1; i < nitems; i++){

		if (items[i] < smallest) return false;

		smallest = items[i];
	}


	return true;

}

__host__ __device__ bool sixteen_byte_assert_sorted(uint16_t * items, uint64_t nitems){


	if (nitems < 1) return true;

	uint16_t smallest = items[0];

	for (int i=1; i < nitems; i++){

		if (items[i] < smallest) return false;

		smallest = items[i];
	}


	return true;

}

__device__ void big_bubble_sort(uint64_t * tags, int fill, int warpID){


	while (true){


		bool sorted = false;

		//even transpositions
		for (int i = warpID*2+1; i < fill; i+=64){

			//swap warpID*2, warpID*2+1

			if ((tags[i-1] & 0xFF) > (tags[i] & 0xFF)){

				uint64_t temp_tag;

				temp_tag = tags[i-1];

				tags[i-1] = tags[i];

				tags[i] = temp_tag;

				sorted = true;

			}



		}


		//odd transpositions
		for (int i = warpID*2+2; i < fill; i+=64){

			//swap warpID*2, warpID*2+1

			if ((tags[i-1] & 0xFF) > (tags[i] & 0xFF)){

				uint64_t temp_tag;

				temp_tag = tags[i-1];

				tags[i-1] = tags[i];

				tags[i] = temp_tag;

				sorted = true;

			}



		}

		if (__ffs(__ballot_sync(0xffffffff, sorted)) == 0) return;


	}


}

__device__ void bubble_sort(uint8_t * tags, int fill, int warpID){




	while (true){


		bool sorted = false;

		//even transpositions
		for (int i = warpID*2+1; i < fill; i+=64){

			//swap warpID*2, warpID*2+1

			if ((tags[i-1] & 0xFF) > (tags[i] & 0xFF)){

				#if TAG_BITS == 8

				uint8_t temp_tag;

				#else

				uint16_t temp_tag;

				#endif

				temp_tag = tags[i-1];

				tags[i-1] = tags[i];

				tags[i] = temp_tag;

				sorted = true;

			}



		}


		//odd transpositions
		for (int i = warpID*2+2; i < fill; i+=64){

			//swap warpID*2, warpID*2+1

			if ((tags[i-1] & 0xFF) > (tags[i] & 0xFF)){

				#if TAG_BITS == 8

				uint8_t temp_tag;

				#else

				uint16_t temp_tag;

				#endif

				temp_tag = tags[i-1];

				tags[i-1] = tags[i];

				tags[i] = temp_tag;

				sorted = true;

			}



		}

		if (__ffs(__ballot_sync(0xffffffff, sorted)) == 0) return;


	}


}


__device__ void merge_dual_arrays(uint8_t * temp_tags, uint8_t * primary, uint8_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID){



	//Primary section - for now fine, but in the future do find_first_below binary search - slightly more 

	__shared__ int primary_counters [WARPS_PER_BLOCK*32];

	__shared__ int secondary_counters [WARPS_PER_BLOCK*32];

	__shared__ int merged_counters[WARPS_PER_BLOCK*32];



	

	primary_counters[teamID*32+warpID] = 0;
	secondary_counters[teamID*32+warpID] = 0;

	merged_counters[teamID*32+warpID] = 0;

	const int dividing_line = 8;




	__syncwarp();


	//approach 3 - binary search for start

	/*
	// int cutoff = dividing_line*warpID; 

	// int lower = 0;

	// int upper = primary_nitems ;


	//approach 2 - partition evenly across
	*/



	//this is overall faster
	//but a binary partition would be even nicer

	int start = warpID*primary_nitems/32;

	int end = (warpID+1)*primary_nitems/32;

	if (warpID == 31) end = primary_nitems;


	for (int i = start; i < end; i++){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}

	start = warpID*secondary_nitems/32;

	end = (warpID+1)*secondary_nitems/32;

	if (warpID == 31) end = secondary_nitems;


	for (int i = start; i < end; i++){

		int index = secondary[i] / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	


	//primary pipeline - add in a row - this condenses adds to the same address
	/*

	for (int i = warpID; i < primary_nitems; i+=32){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}


	for (int i=warpID; i<secondary_nitems; i+=32){

		int index = secondary[i] / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	*/

	__syncwarp();


	int primary_read = primary_counters[teamID*32+warpID];

	int prefix_sum = primary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-primary_read;

	int primary_start = prefix_sum;

	int primary_length = primary_read;


	primary_counters[teamID*32+warpID] = prefix_sum;


	int secondary_read = secondary_counters[teamID*32+warpID];

	prefix_sum = secondary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-secondary_read;

	int secondary_start = prefix_sum;

	int secondary_length = secondary_read;


	//subtracting read gives us an initial start

	secondary_counters[teamID*32+warpID] = prefix_sum;





	//assert(primary_length < 20);
	//assert(secondary_length < 20);


	// int merged_read = merged_counters[teamID*32+warpID];

	// prefix_sum = merged_read;



	// for (int i =1; i<=16; i*=2){

	// 	int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

	// 	if ((warpID) >= i) prefix_sum +=n;

	// }

	// //subtracting read gives us an initial start
	// prefix_sum = prefix_sum-merged_read;

	// int merged_start = prefix_sum;

	// int merged_length = primary_read;


	// merged_counters[teamID*32+warpID] = prefix_sum;

	__syncwarp();

	merged_counters[teamID*32+warpID] = primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


	__syncwarp();

	int merged_start = merged_counters[teamID*32+warpID];


	//need reserved space for entire call - should be nitems?

	//uint8_t * temp_tags = (uint8_t *) malloc(sizeof(uint8_t)*primary_length);


	for (int i = primary_start; i < primary_start+primary_length; i++){

		temp_tags[i] = primary[i];

	}


	//next steps don't need to sync, we can proceed as fast as posssible
	//each thread is independent - may smooth out memload issues
	//syncwarp here for testing



	//after this we have temp_tags[primary_start] -> primary_length
	// secondary[secondary_start] -> secondary_length

	//now to zip from primary[merged_start] -> merged_length

	int primary_end = primary_start + primary_length;

	int secondary_end = secondary_start + secondary_length;

	while (primary_start < primary_end && secondary_start < secondary_end){


		if (temp_tags[primary_start] < secondary[secondary_start]){

			primary[merged_start] = temp_tags[primary_start];

			primary_start++;

		} else {

			primary[merged_start] = secondary[secondary_start];

			secondary_start++;

		}

		merged_start++;


	}

	for (int i = primary_start; i < primary_end; i++){

		primary[merged_start] = temp_tags[i];
		merged_start++;

	}

	for (int i = secondary_start; i < secondary_end; i++){

		primary[merged_start] = secondary[i];
		merged_start++;

	}

	__syncwarp();

	//free(temp_tags);


}


__device__ void merge_dual_arrays_sixteen(uint16_t * temp_tags, uint16_t * primary, uint16_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID){



	//Primary section - for now fine, but in the future do find_first_below binary search - slightly more 

	__shared__ int primary_counters [WARPS_PER_BLOCK*32];

	__shared__ int secondary_counters [WARPS_PER_BLOCK*32];

	__shared__ int merged_counters[WARPS_PER_BLOCK*32];



	

	primary_counters[teamID*32+warpID] = 0;
	secondary_counters[teamID*32+warpID] = 0;

	merged_counters[teamID*32+warpID] = 0;


	//this is different for 8 bit / 16 bit
	//see merge_dual_arrays_16_bit_64_bit for an explanation.
	const int dividing_line = 2048;




	__syncwarp();


	//approach 3 - binary search for start

	/*
	// int cutoff = dividing_line*warpID; 

	// int lower = 0;

	// int upper = primary_nitems ;


	//approach 2 - partition evenly across
	*/



	//this is overall faster
	//but a binary partition would be even nicer

	int start = warpID*primary_nitems/32;

	int end = (warpID+1)*primary_nitems/32;

	if (warpID == 31) end = primary_nitems;


	for (int i = start; i < end; i++){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}

	start = warpID*secondary_nitems/32;

	end = (warpID+1)*secondary_nitems/32;

	if (warpID == 31) end = secondary_nitems;


	for (int i = start; i < end; i++){

		int index = secondary[i] / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	


	//primary pipeline - add in a row - this condenses adds to the same address
	/*

	for (int i = warpID; i < primary_nitems; i+=32){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}


	for (int i=warpID; i<secondary_nitems; i+=32){

		int index = secondary[i] / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	*/

	__syncwarp();


	int primary_read = primary_counters[teamID*32+warpID];

	int prefix_sum = primary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-primary_read;

	int primary_start = prefix_sum;

	int primary_length = primary_read;


	primary_counters[teamID*32+warpID] = prefix_sum;


	int secondary_read = secondary_counters[teamID*32+warpID];

	prefix_sum = secondary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-secondary_read;

	int secondary_start = prefix_sum;

	int secondary_length = secondary_read;


	//subtracting read gives us an initial start

	secondary_counters[teamID*32+warpID] = prefix_sum;





	//assert(primary_length < 20);
	//assert(secondary_length < 20);


	// int merged_read = merged_counters[teamID*32+warpID];

	// prefix_sum = merged_read;



	// for (int i =1; i<=16; i*=2){

	// 	int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

	// 	if ((warpID) >= i) prefix_sum +=n;

	// }

	// //subtracting read gives us an initial start
	// prefix_sum = prefix_sum-merged_read;

	// int merged_start = prefix_sum;

	// int merged_length = primary_read;


	// merged_counters[teamID*32+warpID] = prefix_sum;

	__syncwarp();

	merged_counters[teamID*32+warpID] = primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


	__syncwarp();

	int merged_start = merged_counters[teamID*32+warpID];


	//need reserved space for entire call - should be nitems?

	//uint8_t * temp_tags = (uint8_t *) malloc(sizeof(uint8_t)*primary_length);


	for (int i = primary_start; i < primary_start+primary_length; i++){

		temp_tags[i] = primary[i];

	}


	//next steps don't need to sync, we can proceed as fast as posssible
	//each thread is independent - may smooth out memload issues
	//syncwarp here for testing



	//after this we have temp_tags[primary_start] -> primary_length
	// secondary[secondary_start] -> secondary_length

	//now to zip from primary[merged_start] -> merged_length

	int primary_end = primary_start + primary_length;

	int secondary_end = secondary_start + secondary_length;

	while (primary_start < primary_end && secondary_start < secondary_end){


		if (temp_tags[primary_start] < secondary[secondary_start]){

			primary[merged_start] = temp_tags[primary_start];

			primary_start++;

		} else {

			primary[merged_start] = secondary[secondary_start];

			secondary_start++;

		}

		merged_start++;


	}

	for (int i = primary_start; i < primary_end; i++){

		primary[merged_start] = temp_tags[i];
		merged_start++;

	}

	for (int i = secondary_start; i < secondary_end; i++){

		primary[merged_start] = secondary[i];
		merged_start++;

	}

	__syncwarp();

	//free(temp_tags);


}


__device__ void merge_dual_arrays_8_bit_64_bit(uint8_t * temp_tags, uint8_t * primary, uint64_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID){



	//Primary section - for now fine, but in the future do find_first_below binary search - slightly more 

	// __shared__ int primary_counters [BLOCKS_PER_THREAD_BLOCK*32];

	// __shared__ int secondary_counters [BLOCKS_PER_THREAD_BLOCK*32];

	// __shared__ int merged_counters[BLOCKS_PER_THREAD_BLOCK*32];


	__shared__ int primary_counters [WARPS_PER_BLOCK*32];

	__shared__ int secondary_counters [WARPS_PER_BLOCK*32];

	__shared__ int merged_counters[WARPS_PER_BLOCK*32];

	// #if TAG_BITS == 8

	// //__shared__ uint8_t temp_tags[WARPS_PER_BLOCK*128];

	// #endif


	#if DEBUG_ASSERTS

	assert(teamID < BLOCKS_PER_THREAD_BLOCK);

	assert (teamID < WARPS_PER_BLOCK);

	


	#endif

	

	primary_counters[teamID*32+warpID] = 0;
	secondary_counters[teamID*32+warpID] = 0;

	merged_counters[teamID*32+warpID] = 0;

	const int dividing_line = 8;




	__syncwarp();

	

	int start = warpID*primary_nitems/32;

	int end = (warpID+1)*primary_nitems/32;

	if (warpID == 31) end = primary_nitems;


	for (int i = start; i < end; i++){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}

	start = warpID*secondary_nitems/32;

	end = (warpID+1)*secondary_nitems/32;

	if (warpID == 31) end = secondary_nitems;


	for (int i = start; i < end; i++){

		int index = (secondary[i] & 0xff) / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	


	//outdated assignment method - poor cache locality
	/*

	for (int i = warpID; i < primary_nitems; i+=32){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}


	for (int i=warpID; i<secondary_nitems; i+=32){

		int index = (secondary[i] & 0xff) / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	*/
	

	__syncwarp();


	int primary_read = primary_counters[teamID*32+warpID];

	int prefix_sum = primary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-primary_read;

	int primary_start = prefix_sum;

	int primary_length = primary_read;


	primary_counters[teamID*32+warpID] = prefix_sum;


	int secondary_read = secondary_counters[teamID*32+warpID];

	prefix_sum = secondary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-secondary_read;

	int secondary_start = prefix_sum;

	int secondary_length = secondary_read;


	//subtracting read gives us an initial start

	secondary_counters[teamID*32+warpID] = prefix_sum;





	//assert(primary_length < 20);
	//assert(secondary_length < 20);


	// int merged_read = merged_counters[teamID*32+warpID];

	// prefix_sum = merged_read;



	// for (int i =1; i<=16; i*=2){

	// 	int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

	// 	if ((warpID) >= i) prefix_sum +=n;

	// }

	// //subtracting read gives us an initial start
	// prefix_sum = prefix_sum-merged_read;

	// int merged_start = prefix_sum;

	// int merged_length = primary_read;


	// merged_counters[teamID*32+warpID] = prefix_sum;

	__syncwarp();

	merged_counters[teamID*32+warpID] = primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


	__syncwarp();

	int merged_start = merged_counters[teamID*32+warpID];



	

	//need reserved space for entire call - should be nitems?



	//uint8_t * temp_tags = (uint8_t *) malloc(sizeof(uint8_t)*primary_length);

	

	for (int i = primary_start; i < primary_start+primary_length; i++){

		temp_tags[i] = primary[i];

	}


	__syncwarp();



	//next steps don't need to sync, we can proceed as fast as posssible
	//each thread is independent - may smooth out memload issues
	//syncwarp here for testing



	//after this we have temp_tags[primary_start] -> primary_length
	// secondary[secondary_start] -> secondary_length

	//now to zip from primary[merged_start] -> merged_length

	int primary_end = primary_start + primary_length;

	int secondary_end = secondary_start + secondary_length;

	while (primary_start < primary_end && secondary_start < secondary_end){


		if (temp_tags[primary_start] < (secondary[secondary_start] & 0xff)){

			primary[merged_start] = temp_tags[primary_start];

			primary_start++;

		} else {

			primary[merged_start] = secondary[secondary_start] & 0xff;

			secondary_start++;

		}

		merged_start++;


	}

	for (int i = primary_start; i < primary_end; i++){

		primary[merged_start] = temp_tags[i];
		merged_start++;

	}

	for (int i = secondary_start; i < secondary_end; i++){

		primary[merged_start] = secondary[i] & 0xff;
		merged_start++;

	}

	__syncwarp();

	



	//free(temp_tags);


}


__device__ void merge_3_into_tags_16_bit(uint16_t * tags, uint64_t * global_buffer, int buffer_count, uint16_t * primary, int primary_nitems, uint16_t * secondary, int secondary_nitems, int teamID, int warpID){



	//Primary section - for now fine, but in the future do find_first_below binary search - slightly more 

	// __shared__ int primary_counters [BLOCKS_PER_THREAD_BLOCK*32];

	// __shared__ int secondary_counters [BLOCKS_PER_THREAD_BLOCK*32];

	// __shared__ int merged_counters[BLOCKS_PER_THREAD_BLOCK*32];


	__shared__ int buffer_counters [WARPS_PER_BLOCK*32];

	__shared__ int primary_counters [WARPS_PER_BLOCK*32];

	__shared__ int secondary_counters [WARPS_PER_BLOCK*32];

	__shared__ int merged_counters[WARPS_PER_BLOCK*32];

	// #if TAG_BITS == 8

	// //__shared__ uint8_t temp_tags[WARPS_PER_BLOCK*128];

	// #endif


	#if DEBUG_ASSERTS

	assert(teamID < BLOCKS_PER_THREAD_BLOCK);

	assert (teamID < WARPS_PER_BLOCK);


	assert(two_byte_assert_sorted(global_buffer, buffer_count));


	assert(sixteen_byte_assert_sorted(primary, primary_nitems));
	assert(sixteen_byte_assert_sorted(secondary, secondary_nitems));
	


	#endif


	buffer_counters[teamID*32+warpID] = 0;	

	primary_counters[teamID*32+warpID] = 0;
	secondary_counters[teamID*32+warpID] = 0;

	//merged_counters[teamID*32+warpID] = 0;


	//dividing line is MAX_VAL/32;

	//for 8 bits this is 256/32

	//16 bits is 65536/32 = 2048;

	const int dividing_line = 2048;




	__syncwarp();


	int start = warpID*primary_nitems/32;

	int end = (warpID+1)*primary_nitems/32;

	if (warpID == 31) end = primary_nitems;


	for (int i = start; i < end; i++){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}

	start = warpID*secondary_nitems/32;

	end = (warpID+1)*secondary_nitems/32;

	if (warpID == 31) end = secondary_nitems;


	for (int i = start; i < end; i++){

		int index = (secondary[i]) / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	start = warpID*buffer_count/32;

	end = (warpID+1)*buffer_count/32;

	if (warpID == 31) end = buffer_count;


	for (int i = start; i < end; i++){

		int index = (global_buffer[i] & 0xffff) / dividing_line;

		atomicAdd(& buffer_counters[teamID*32 + index], 1);


	}

	//outdated primary assignment method

	/*

	for (int i = warpID; i < primary_nitems; i+=32){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}


	for (int i=warpID; i<secondary_nitems; i+=32){

		int index = (secondary[i] & 0xffff) / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	*/

	__syncwarp();


	int primary_read = primary_counters[teamID*32+warpID];

	int prefix_sum = primary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-primary_read;

	int primary_start = prefix_sum;

	int primary_length = primary_read;


	primary_counters[teamID*32+warpID] = prefix_sum;


	//secondary sync


	int secondary_read = secondary_counters[teamID*32+warpID];

	prefix_sum = secondary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-secondary_read;

	int secondary_start = prefix_sum;

	int secondary_length = secondary_read;


	//subtracting read gives us an initial start

	secondary_counters[teamID*32+warpID] = prefix_sum;



	//buffer sync

	int buffer_read = buffer_counters[teamID*32+warpID];

	prefix_sum = buffer_read;


	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-buffer_read;

	int buffer_start = prefix_sum;

	int buffer_length = buffer_read;

	buffer_counters[teamID*32+warpID] = prefix_sum;

	//assert(primary_length < 20);
	//assert(secondary_length < 20);


	// int merged_read = merged_counters[teamID*32+warpID];

	// prefix_sum = merged_read;



	// for (int i =1; i<=16; i*=2){

	// 	int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

	// 	if ((warpID) >= i) prefix_sum +=n;

	// }

	// //subtracting read gives us an initial start
	// prefix_sum = prefix_sum-merged_read;

	// int merged_start = prefix_sum;

	// int merged_length = primary_read;


	// merged_counters[teamID*32+warpID] = prefix_sum;

	__syncwarp();

	//merged_counters[teamID*32+warpID] = buffer_counters[teamID*32+warpID] + primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


	//__syncwarp();

	int merged_start = buffer_counters[teamID*32+warpID] + primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


	

	//uint8_t * temp_tags = (uint8_t *) malloc(sizeof(uint8_t)*primary_length);

	

	// for (int i = primary_start; i < primary_start+primary_length; i++){

	// 	temp_tags[i] = primary[i];

	// }


	// #if DEBUG_ASSERTS


	// assert(sixteen_byte_assert_sorted(temp_tags + primary_start, primary_length));


	// #endif


	//__syncwarp();


	// #if DEBUG_ASSERTS

	// for (int i = primary_start; i < primary_start+primary_length; i++){

	// 	assert(primary[i] == temp_tags[i]);

	// }

	// __syncwarp();


	// #endif

	//next steps don't need to sync, we can proceed as fast as posssible
	//each thread is independent - may smooth out memload issues
	//syncwarp here for testing

	//return here just to double triple check
	

	//rewriting this to be simpler


	int primary_end = primary_start + primary_length;
	int secondary_end = secondary_start + secondary_length;
	int buffer_end = buffer_start + buffer_length;


	bool buffer_ended = buffer_start >= buffer_end;
	bool primary_ended = primary_start >= primary_end;
	bool secondary_ended = secondary_start >= secondary_end;


	while (!buffer_ended || !primary_ended || !secondary_ended){

		#if DEBUG_ASSERTS

		if (primary_start >= primary_end) assert(primary_ended);
		if (secondary_start >= secondary_end) assert(secondary_ended);
		if (buffer_start >= buffer_end) assert(buffer_ended);

		#endif


		if (!buffer_ended && (secondary_ended || (global_buffer[buffer_start] & 0xffff) <= secondary[secondary_start]) && (primary_ended || (global_buffer[buffer_start] & 0xffff) <= primary[primary_start])){


			tags[merged_start] = global_buffer[buffer_start] & 0xffff;
			buffer_start++;

		} else if (!primary_ended && (buffer_ended || primary[primary_start] <= (global_buffer[buffer_start] & 0xffff)) && (secondary_ended || primary[primary_start] <= secondary[secondary_start])){

			tags[merged_start] = primary[primary_start];
			primary_start++;

		} else {

			tags[merged_start] = secondary[secondary_start];
			secondary_start++;
		}




		merged_start++;

		buffer_ended = buffer_start >= buffer_end;
		primary_ended = primary_start >= primary_end;
		secondary_ended = secondary_start >= secondary_end;

	}


	__syncwarp();



	//free(temp_tags);


}


__device__ void merge_dual_arrays_16_bit_64_bit(uint16_t * temp_tags, uint16_t * primary, uint64_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID){



	//Primary section - for now fine, but in the future do find_first_below binary search - slightly more 

	// __shared__ int primary_counters [BLOCKS_PER_THREAD_BLOCK*32];

	// __shared__ int secondary_counters [BLOCKS_PER_THREAD_BLOCK*32];

	// __shared__ int merged_counters[BLOCKS_PER_THREAD_BLOCK*32];


	__shared__ int primary_counters [WARPS_PER_BLOCK*32];

	__shared__ int secondary_counters [WARPS_PER_BLOCK*32];

	__shared__ int merged_counters[WARPS_PER_BLOCK*32];

	// #if TAG_BITS == 8

	// //__shared__ uint8_t temp_tags[WARPS_PER_BLOCK*128];

	// #endif


	#if DEBUG_ASSERTS

	assert(teamID < BLOCKS_PER_THREAD_BLOCK);

	assert (teamID < WARPS_PER_BLOCK);


	assert(sixteen_byte_assert_sorted(primary, primary_nitems));

	assert(two_byte_assert_sorted(secondary, secondary_nitems));
	


	#endif

	

	primary_counters[teamID*32+warpID] = 0;
	secondary_counters[teamID*32+warpID] = 0;

	merged_counters[teamID*32+warpID] = 0;


	//dividing line is MAX_VAL/32;

	//for 8 bits this is 256/32

	//16 bits is 65536/32 = 2048;

	const int dividing_line = 2048;




	__syncwarp();


	int start = warpID*primary_nitems/32;

	int end = (warpID+1)*primary_nitems/32;

	if (warpID == 31) end = primary_nitems;


	for (int i = start; i < end; i++){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}

	start = warpID*secondary_nitems/32;

	end = (warpID+1)*secondary_nitems/32;

	if (warpID == 31) end = secondary_nitems;


	for (int i = start; i < end; i++){

		int index = (secondary[i] & 0xffff) / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	//outdated primary assignment method

	/*

	for (int i = warpID; i < primary_nitems; i+=32){

		int index = primary[i] / dividing_line;

		atomicAdd(& primary_counters[teamID*32 + index], 1);


	}


	for (int i=warpID; i<secondary_nitems; i+=32){

		int index = (secondary[i] & 0xffff) / dividing_line;

		atomicAdd(& secondary_counters[teamID*32 + index], 1);


	}

	*/

	__syncwarp();


	int primary_read = primary_counters[teamID*32+warpID];

	int prefix_sum = primary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-primary_read;

	int primary_start = prefix_sum;

	int primary_length = primary_read;


	primary_counters[teamID*32+warpID] = prefix_sum;


	int secondary_read = secondary_counters[teamID*32+warpID];

	prefix_sum = secondary_read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-secondary_read;

	int secondary_start = prefix_sum;

	int secondary_length = secondary_read;


	//subtracting read gives us an initial start

	secondary_counters[teamID*32+warpID] = prefix_sum;





	//assert(primary_length < 20);
	//assert(secondary_length < 20);


	// int merged_read = merged_counters[teamID*32+warpID];

	// prefix_sum = merged_read;



	// for (int i =1; i<=16; i*=2){

	// 	int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

	// 	if ((warpID) >= i) prefix_sum +=n;

	// }

	// //subtracting read gives us an initial start
	// prefix_sum = prefix_sum-merged_read;

	// int merged_start = prefix_sum;

	// int merged_length = primary_read;


	// merged_counters[teamID*32+warpID] = prefix_sum;

	__syncwarp();

	merged_counters[teamID*32+warpID] = primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


	__syncwarp();

	int merged_start = merged_counters[teamID*32+warpID];



	

	//need reserved space for entire call - should be nitems?



	//uint8_t * temp_tags = (uint8_t *) malloc(sizeof(uint8_t)*primary_length);

	

	for (int i = primary_start; i < primary_start+primary_length; i++){

		temp_tags[i] = primary[i];

	}


	#if DEBUG_ASSERTS


	assert(sixteen_byte_assert_sorted(temp_tags + primary_start, primary_length));


	#endif


	__syncwarp();


	#if DEBUG_ASSERTS

	for (int i = primary_start; i < primary_start+primary_length; i++){

		assert(primary[i] == temp_tags[i]);

	}

	__syncwarp();


	#endif

	//next steps don't need to sync, we can proceed as fast as posssible
	//each thread is independent - may smooth out memload issues
	//syncwarp here for testing



	//after this we have temp_tags[primary_start] -> primary_length
	// secondary[secondary_start] -> secondary_length

	//now to zip from primary[merged_start] -> merged_length

	int primary_end = primary_start + primary_length;

	int secondary_end = secondary_start + secondary_length;

	while (primary_start < primary_end && secondary_start < secondary_end){


		if (temp_tags[primary_start] < (secondary[secondary_start] & 0xffff)){

			primary[merged_start] = temp_tags[primary_start];

			primary_start++;

		} else {

			primary[merged_start] = secondary[secondary_start] & 0xffff;

			secondary_start++;

		}

		merged_start++;


	}

	for (int i = primary_start; i < primary_end; i++){

		primary[merged_start] = temp_tags[i];
		merged_start++;

	}

	for (int i = secondary_start; i < secondary_end; i++){

		primary[merged_start] = secondary[i] & 0xffff;
		merged_start++;

	}

	__syncwarp();



	#if DEBUG_ASSERTS


	//have each thread check their segment

	assert(sixteen_byte_assert_sorted(primary + merged_counters[teamID*32+warpID], primary_length+secondary_length));


	assert(sixteen_byte_assert_sorted(primary, primary_nitems+secondary_nitems));


	#endif

	



	//free(temp_tags);


}



__device__ void short_warp_sort(uint8_t * items, int nitems, int teamID, int warpID){


	__shared__ int counters [WARPS_PER_BLOCK*32];

	#if TAG_BITS == 8


	__shared__ uint8_t temp_tags [WARPS_PER_BLOCK*32];

	__shared__ uint8_t alt_temp_tags [WARPS_PER_BLOCK*32];


	#elif TAG_BITS == 16

	__shared__ uint16_t temp_tags [WARPS_PER_BLOCK*32];

	__shared__ uint16_t alt_temp_tags [WARPS_PER_BLOCK*32];

	#endif




	counters[teamID*32+warpID] = 0;

	__syncwarp();


	//need to split the upper bits into divisible regions, divide by MAX/32?
	//256/32 == 8

	const int dividing_line = 8;


	for (int i = warpID; i < nitems; i+=32){

		int index = items[i] / dividing_line;

		atomicAdd(& counters[teamID*32 + index], 1);
	}

	__syncwarp();


	int read = counters[teamID*32+warpID];

	int prefix_sum = read;



	for (int i =1; i<=16; i*=2){

		int n = __shfl_up_sync(0xffffffff, prefix_sum, i, 32);

		if ((warpID) >= i) prefix_sum +=n;

	}

	//subtracting read gives us an initial start
	prefix_sum = prefix_sum-read;

	int start = prefix_sum;

	int length = read;


	counters[teamID*32+warpID] = prefix_sum;


	__syncwarp();


	//use read as a stopping condition

	temp_tags[teamID*32+warpID] = items[counters[teamID*32+warpID]];

	while (true){

		int ballot = 0;

		if (read >= 0){


			//grab available slot
			ballot = 1;


			int index = temp_tags[teamID*32+warpID] / dividing_line;  

			int index_to_write = atomicAdd(&counters[teamID*32+index], 1);


			

			//this is going to break on the last index

			if (index_to_write + 1 < nitems)
			alt_temp_tags[teamID*32+warpID] = items[index_to_write+1];



			__syncwarp();


			items[index_to_write] = temp_tags[teamID*32+warpID];

			temp_tags[teamID*32+warpID] = alt_temp_tags[teamID*32+warpID];

			read--;
		} else {

			__syncwarp();
		}



		//if no threads - no one is working anymore - end
		if (!__ffs(__ballot_sync(0xffffffff, ballot))) break;


	}


	__syncwarp();

	//at this point each mini section contains the items, 

	//do insertion sort

	//we need - info 

	//start - start + length;

	uint8_t min = items[start];

	for (int i = start+1; i < start+length; i++){


		for (int j = i; j < start+length; j++){


			//swap with smaller
			if (items[j] < min){

				uint8_t temp = items[j];
				items[j] = min;
				min = temp;

			}


		}


		items[i-1] = min;



	}




	__syncwarp();


		





}

//handled by one thread atm
__device__ void insertion_sort_max(uint8_t * items, int nitems){


	for (int i = nitems-1; i >= 32; i--){

		uint8_t max = 0;

		int max_index = 0;

		//loop through all items beow the established index
		for (int j=0; j <= i; j++){


			if (items[j] > max){

				max = items[j];

				max_index = j;
			}


		}


		//swap i and max_index

		//max is already set to items[j], save a cycle
		items[max_index] = items[i];
		items[i] = max;





	}



}

//handled by one thread atm
__device__ void insertion_sort_16(uint16_t * items, int nitems){


	for (int i = nitems-1; i >= 32; i--){

		uint16_t max = 0;

		int max_index = 0;

		//loop through all items beow the established index
		for (int j=0; j <= i; j++){


			if (items[j] > max){

				max = items[j];

				max_index = j;
			}


		}


		//swap i and max_index

		//max is already set to items[j], save a cycle
		items[max_index] = items[i];
		items[i] = max;





	}



}

__device__ void sorting_network_8_bit(uint8_t * items, int nitems, int warpID){


	//this implementation uses batcher odd-even mergesort


	//implementation 2 - massively parallel radix sort

	// uint8_t * main_buffer = items;

	// uint8_t * alt_buffer = items + 64;


	//calculate bit
	for (int mask_bit = 0; mask_bit < 8; mask_bit++){

		uint8_t mask = 1 << mask_bit;


		bool my_bit = false;


		if (warpID < nitems){

			my_bit = items[warpID] & mask;
		}



		


		//and ballot synch on this bit
		unsigned int result = __ballot_sync(0xffffffff, my_bit);

		//histogram

		//TODO: double check me - this "histogram" calc should prevent incorrect items from summing

		//int num_ones = __popc(result);

		int num_zeros = nitems - __popc(result);


		//prefix sum of zeros

		int zero_sum = !my_bit;



		for (int i =1; i<=16; i*=2){

			int n = __shfl_up_sync(0xffffffff, zero_sum, i, 32);

			if ((warpID) >= i) zero_sum +=n;

		}

		//subtracting read gives us an initial start
		zero_sum = zero_sum - !my_bit;



		int one_sum = my_bit;

		for (int i =1; i<=16; i*=2){

			int n = __shfl_up_sync(0xffffffff, one_sum, i, 32);

			if ((warpID) >= i) one_sum +=n;

		}

		//subtracting read gives us an initial start
		one_sum = one_sum - my_bit;


		//__syncwarp();

		uint8_t my_item = items[warpID];

		__syncwarp();

		if (warpID < nitems){

				

			//if one bit(num_zeros + one_sum);
			//if zero !bit(zero_sum)

			int write_index = zero_sum;

			if (my_bit){

				write_index = num_zeros+one_sum;

			}


			items[ write_index ] = my_item;


		}

		__syncwarp();


		// my_bit = items[warpID] & mask;
		// //and ballot synch on this bit
		// result = __ballot_sync(0xffffffff, my_bit);

		__syncwarp();



	}




}


__device__ void sorting_network_16_bit(uint16_t * items, int nitems, int warpID){


	//this implementation uses batcher odd-even mergesort


	//implementation 2 - massively parallel radix sort

	// uint8_t * main_buffer = items;

	// uint8_t * alt_buffer = items + 64;


	//calculate bit
	for (int mask_bit = 0; mask_bit < 16; mask_bit++){

		uint16_t mask = 1 << mask_bit;


		bool my_bit = false;


		if (warpID < nitems){

			my_bit = items[warpID] & mask;
		}



		


		//and ballot synch on this bit
		unsigned int result = __ballot_sync(0xffffffff, my_bit);

		//histogram

		//TODO: double check me - this "histogram" calc should prevent incorrect items from summing

		//int num_ones = __popc(result);

		int num_zeros = nitems - __popc(result);


		//prefix sum of zeros

		int zero_sum = !my_bit;



		for (int i =1; i<=16; i*=2){

			int n = __shfl_up_sync(0xffffffff, zero_sum, i, 32);

			if ((warpID) >= i) zero_sum +=n;

		}

		//subtracting read gives us an initial start
		zero_sum = zero_sum - !my_bit;



		int one_sum = my_bit;

		for (int i =1; i<=16; i*=2){

			int n = __shfl_up_sync(0xffffffff, one_sum, i, 32);

			if ((warpID) >= i) one_sum +=n;

		}

		//subtracting read gives us an initial start
		one_sum = one_sum - my_bit;


		//__syncwarp();

		uint16_t my_item;

		if (warpID < nitems)

			my_item = items[warpID];

		__syncwarp();

		if (warpID < nitems){

				

			//if one bit(num_zeros + one_sum);
			//if zero !bit(zero_sum)

			int write_index = zero_sum;

			if (my_bit){

				write_index = num_zeros+one_sum;

			}


			items[ write_index ] = my_item;


		}

		__syncwarp();


		// my_bit = items[warpID] & mask;
		// //and ballot synch on this bit
		// result = __ballot_sync(0xffffffff, my_bit);

		__syncwarp();



	}




}


#endif