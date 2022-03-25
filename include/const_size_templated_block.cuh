#ifndef CONST_SIZE_TEMPLATED_BLOCK_H 
#define CONST_SIZE_TEMPLATED_BLOCK_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/metadata.cuh"
#include "include/key_val_pair.cuh"
#include <stdio.h>
#include <assert.h>




#define BYTES_AVAILABLE (BYTES_PER_CACHE_LINE * CACHE_LINES_PER_BLOCK)

#define dim(x) (sizeof(x) / sizeof((x)[0]))


//counters are now external to allow them to permanently reside in the l1 cache.
//this should improve performance and allow for different loading schemes
//that are less reliant on the initial load.

//these are templated on just one thing
//key_value_pairs



template <typename Tag_type>
__device__ bool assert_sorted(Tag_type * tags, int nitems){


	if (nitems < 1) return true;

	Tag_type smallest = tags[0];

	for (int i=1; i< nitems; i++){

		if (tags[i] < smallest){

			int bad_index = i;
			printf("Bad index: %d/%d\n", bad_index, nitems);
			return false;

		}

		smallest = tags[i];
	}

	return true;

}



template <typename Tag_type>
struct __attribute__ ((__packed__)) templated_block {


	//tag bits change based on the #of bytes allocated per block



	Tag_type tags[SLOTS_PER_CONST_BLOCK];



	int print_max_size(){

		return SLOTS_PER_CONST_BLOCK;

	};

	__host__ __device__ static int max_size(){

		return SLOTS_PER_CONST_BLOCK;
	}

	__host__ __device__ static int fill_cutoff(){

		return SLOTS_PER_CONST_BLOCK*.75;
	}

 

 	__device__ bool dump_single_item(Tag_type item, int warpID){


 		#if DEBUG_ASSERTS

 			assert(max_size() <= 32);

 		#endif

 		bool open = false;

 		if (warpID < max_size() && tags[warpID].get_key() == 0){

 			open = true;
 		}


 		unsigned int ballot_result = __ballot_sync(0xffffffff, open);


		int first_thread = __ffs(ballot_result)-1;

		if (first_thread == warpID){

			tags[warpID] = item;
		}

		__threadfence();
		//only 0 if no one inserted
		return first_thread+1;



 	}


 	__device__ bool dump_single_item_reserved_slot(Tag_type item, int reserved_slot, int warpID){


 		if (warpID == 0){

 			#if DEBUG_ASSERTS
 			assert(tags[reserved_slot].get_key() == 0);
 			#endif

 			tags[reserved_slot] = item;
 		}

 		__threadfence();

 		return true;
 	}

 	// __device__ bool dump_single_item_assert_write(Tag_type item, int warpID){

 	// 	#if DEBUG_ASSERTS

 	// 		assert(max_size() <= 32);

 	// 	#endif

 	// 	bool inserted = false;

 	// 	while (!inserted){

 	// 		bool open = false;

 	// 		if (warpID < max_size() && tags[warpID].get_key() == 0)
 	// 	}

 	// }


 	__device__ bool dump_multiple_items(Tag_type * items, int nitems, int warpID){

 		#if DEBUG_ASSERTS

 			assert(max_size() <= 32);

 		#endif

 		bool open = false;

 		if (warpID < max_size() && tags[warpID].get_key() == 0){

 			open = true;
 		}

 		unsigned int ballot_result = __ballot_sync(0xffffffff, open);

 		//get mask and popcount

 		//compress mask from 64 bits so that 32 works as intended
 		unsigned int mask = (2ULL << warpID) -1;

 		int my_item = __popc(ballot_result & mask) -1;

 		if (open && my_item < nitems){

 			tags[warpID] = items[my_item];
 		}


 		return true;

 	}


 	__device__ Tag_type query_single_item(Tag_type item, int warpID, bool & wasFound){


 		#if DEBUG_ASSERTS

 			assert(max_size() <= 32);

 		#endif


 		//perform a ballot
 		bool found = false;

 		if (warpID< max_size() && tags[warpID].get_key() == item.get_key()){


 			found = true;

 		}

 		unsigned int ballot_result = __ballot_sync(0xffffffff, found);


		int first_thread = __ffs(ballot_result)-1;

		wasFound = (first_thread+1);

		if (!wasFound) return tags[0];

		return tags[first_thread];


 	}



	__device__ void dump_all_buffers_sorted(Tag_type * global_buffer, int buffer_count, Tag_type * primary, int primary_nitems, Tag_type * secondary, const int secondary_nitems, const int teamID, const int warpID, uint64_t dividing_line){

		__shared__ int buffer_counters [WARPS_PER_BLOCK*32];

		__shared__ int primary_counters [WARPS_PER_BLOCK*32];

		__shared__ int secondary_counters [WARPS_PER_BLOCK*32];

		//__shared__ int merged_counters[WARPS_PER_BLOCK*32];

		//start of merge


		buffer_counters[teamID*32+warpID] = 0;	

		primary_counters[teamID*32+warpID] = 0;
		secondary_counters[teamID*32+warpID] = 0;

		//merged_counters[teamID*32+warpID] = 0;


		//dividing line is MAX_VAL/32;

		//for 8 bits this is 256/32

		//16 bits is 65536/32 = 2048;


		//use this signal if you 
		//const uint64_t dividing_line = 1ULL << 8*sizeof(((Tag_type){0}).get_key);
		//check if i can bring this back

		__syncwarp();

		int start = warpID*primary_nitems/32;

		int end = (warpID+1)*primary_nitems/32;

		if (warpID == 31) end = primary_nitems;


		for (int i = start; i < end; i++){

			int index = primary[i] / dividing_line;

			#if DEBUG_ASSERTS

			assert(teamID*32 + index < 32*WARPS_PER_BLOCK);

			#endif

			atomicAdd(& primary_counters[teamID*32 + index], 1);


		}

		start = warpID*secondary_nitems/32;

		end = (warpID+1)*secondary_nitems/32;

		if (warpID == 31) end = secondary_nitems;


		for (int i = start; i < end; i++){

			int index = (secondary[i]) / dividing_line;

			#if DEBUG_ASSERTS

			assert(teamID*32 + index < 32*WARPS_PER_BLOCK);
			
			#endif

			atomicAdd(& secondary_counters[teamID*32 + index], 1);


		}

		start = warpID*buffer_count/32;

		end = (warpID+1)*buffer_count/32;

		if (warpID == 31) end = buffer_count;


		for (int i = start; i < end; i++){

			int index = (global_buffer[i]) / dividing_line;

			#if DEBUG_ASSERTS

			assert(teamID*32 + index < 32*WARPS_PER_BLOCK);
			
			#endif

			atomicAdd(& buffer_counters[teamID*32 + index], 1);


		}


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

		__syncwarp();


		int merged_start = buffer_counters[teamID*32+warpID] + primary_counters[teamID*32+warpID] + secondary_counters[teamID*32+warpID];


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


			if (!buffer_ended && (secondary_ended || (global_buffer[buffer_start]) <= secondary[secondary_start]) && (primary_ended || (global_buffer[buffer_start]) <= primary[primary_start])){


				tags[merged_start] = global_buffer[buffer_start];
				buffer_start++;

			} else if (!primary_ended && (buffer_ended || primary[primary_start] <= (global_buffer[buffer_start])) && (secondary_ended || primary[primary_start] <= secondary[secondary_start])){

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




		//end of merge


		//assert sorted at the end
		#if DEBUG_ASSERTS

		assert(assert_sorted<Tag_type>(tags, buffer_count+primary_nitems+secondary_nitems));

		#endif

		return;



	}


	__device__ void sorted_bulk_query(int tag_count, int warpID, Tag_type * items, bool * found, uint64_t nitems){


		#if DEBUG_ASSERTS

		assert(assert_sorted<Tag_type>(tags, tag_count));
		assert(assert_sorted<Tag_type>(items, nitems));

		#endif

		if (tag_count == 0 || nitems == 0) return;

		int left = 0;
		int right = 0;


		while (true){



			if (items[left] == tags[right]){

				found[left] = true;
				left++;

				if (left>=nitems) return;

			} else if (items[left] < tags[right]){
				found[left] = false;
				left++;

				if (left>=nitems) return;
			} else {


				right ++;

				if (right >= tag_count){


					for (int i = left; i < nitems; i++){

						found[i] = false;
					}

					return;
				}
			}
		}


	}
	


	__device__ bool binary_search_query(Tag_type item, int fill){


		#if DEBUG_ASSERTS



		assert(assert_sorted<Tag_type>(tags, fill));


		#endif


		int lower = 0;
		int upper = fill;

		int index;

		while (upper != lower){

		index = lower + (upper - lower)/2;


		Tag_type query_item = tags[index];

		if (query_item < item){

			lower = index+1;

		} else if (query_item > item){

			upper = index;

		} else {

			return true;
		}


		}

		if (lower < fill && tags[lower] == item) return true;

		return false;



	}



};


template<typename Tag_type>
__device__ bool blocks_equal(templated_block<Tag_type> first, templated_block<Tag_type> second, int size){


	bool equal = true;
	for (int i =0; i < size; i++){

		if (first.tags[i] != second.tags[i]) equal = false;

	}

	return equal;

}


#endif //GPU_BLOCK_