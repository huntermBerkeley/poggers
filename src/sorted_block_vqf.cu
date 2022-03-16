#ifndef SORTED_BLOCK_VQF_C
#define SORTED_BLOCK_VQF_C


#include <cuda.h>
#include <cuda_runtime_api.h>


#include "include/sorted_block_vqf.cuh"
#include "include/atomic_block.cuh"
#include "include/hashutil.cuh"
#include "include/metadata.cuh"

#include <iostream>

#include <fstream>
#include <assert.h>

//Thrust Sorting
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include <chrono>
#include <iostream>


#include "include/sorting_helper.cuh"

 

__device__ void optimized_vqf::lock_block(int warpID, uint64_t team, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 0,1) != 0);	
	// }
	// __syncwarp();

	//TODO: turn me back on

	#if EXCLUSIVE_ACCESS


	#else 
	blocks[team].internal_blocks[lock].lock(warpID);

	#endif
}

__device__ void optimized_vqf::unlock_block(int warpID, uint64_t team,  uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 1,0) != 1);	

	// }

	// __syncwarp();

	#if EXCLUSIVE_ACCESS


	#else

	blocks[team].internal_blocks[lock].unlock(warpID);

	#endif


}

__device__ void optimized_vqf::lock_blocks(int warpID, uint64_t team1, uint64_t lock1, uint64_t team2, uint64_t lock2){


	if (team1 * WARPS_PER_BLOCK + lock1 < team2 * WARPS_PER_BLOCK + lock2){

		lock_block(warpID, team1, lock1);
		lock_block(warpID, team2, lock2);
		//while(atomicCAS(locks + lock2, 0,1) == 1);

	} else {


		lock_block(warpID, team2, lock2);
		lock_block(warpID, team1, lock1);
		
	}

	


}

__device__ void optimized_vqf::unlock_blocks(int warpID, uint64_t team1, uint64_t lock1, uint64_t team2, uint64_t lock2){


	if (team1 * WARPS_PER_BLOCK + lock1 < team2 * WARPS_PER_BLOCK + lock2){

		unlock_block(warpID, team1, lock1);
		unlock_block(warpID, team2, lock2);
		//while(atomicCAS(locks + lock2, 0,1) == 1);

	} else {


		unlock_block(warpID, team2, lock2);
		unlock_block(warpID, team1, lock1);
		
	}

	


}

// __device__ bool optimized_vqf::insert(int warpID, uint64_t key, bool hashed){


// 	uint64_t hash;

// 	if (hashed){

// 		hash = key;

// 	} else {

// 		hash = hash_key(key);


// 	}

//    uint64_t block_index = get_bucket_from_hash(hash);

//    //uint64_t alt_block_index = get_alt_hash(hash, block_index);



//  	lock_block(warpID, block_index);

//    int fill_main = blocks[block_index].get_fill();



//    bool toReturn = false;


//    	if (fill_main < MAX_FILL){
//    		blocks[block_index].insert(warpID, hash);

//    		toReturn = true;


//    		#if DEBUG_ASSERTS
//    		int new_fill = blocks[block_index].get_fill();
//    		if (new_fill != fill_main+1){

//    		//blocks[block_index].printMetadata();
//    		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
//    		assert(blocks[block_index].get_fill() == fill_main+1);
//    		}

//    		assert(blocks[block_index].query(warpID, hash));
//    		#endif

//    	}

//    unlock_block(warpID, block_index);



//    return toReturn;





// }


//global call to create thread groups and trigger inserts;
//this is done inside of block_vqf.cu so that cg only needs to be brought in once
__global__ void bulk_insert_kernel(optimized_vqf * vqf, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	uint64_t teamID = tid / (BLOCK_SIZE);



	//TODO double check me
	if (teamID >= vqf->num_teams) return;


	vqf->mini_filter_block(misses);

	return;

	


}

//global call to create thread groups and trigger inserts;
//this is done inside of block_vqf.cu so that cg only needs to be brought in once
__global__ void sorted_bulk_insert_kernel(optimized_vqf * vqf, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	uint64_t teamID = tid / (BLOCK_SIZE);



	//TODO double check me
	if (teamID >= vqf->num_teams) return;


	//vqf->sorted_mini_filter_block(misses);

	vqf->sorted_mini_filter_block_async_write(misses);


	return;

	


}

__global__ void bulk_query_kernel(optimized_vqf * vqf, uint64_t * items, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t teamID = tid / (BLOCK_SIZE);


	if (teamID >= vqf->num_teams) return;

	vqf->mini_filter_queries(items, hits);
}


__global__ void bulk_sorted_query_kernel(optimized_vqf * vqf, uint64_t * items, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t teamID = tid / (BLOCK_SIZE);

	#if DEBUG_ASSERTS

	assert(teamID == blockIdx.x);

	#endif

	if (teamID >= vqf->num_teams) return;

	vqf->mini_filter_bulk_queries(items, hits);
}



//attach buffers, create thread groups, and launch
__host__ void optimized_vqf::bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses){



	uint64_t num_teams = get_num_teams();


	attach_buffers(items, nitems);

	bulk_insert_kernel<<<num_teams, BLOCK_SIZE>>>(this, misses);



}


//attach buffers, create thread groups, and launch
__host__ void optimized_vqf::sorted_bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses){



	uint64_t num_teams = get_num_teams();


	attach_buffers(items, nitems);

	sorted_bulk_insert_kernel<<<num_teams, BLOCK_SIZE>>>(this, misses);



}


__host__ void optimized_vqf::sorted_bulk_insert_buffers_preattached(uint64_t * misses){


	uint64_t num_teams = get_num_teams();

	sorted_bulk_insert_kernel<<<num_teams, BLOCK_SIZE>>>(this, misses);

}


//speed up queries by sorting to minimize memory movements required.
// This will sort the query list but won't mutate the items themselves
__host__ void optimized_vqf::bulk_query(uint64_t * items, uint64_t nitems, bool * hits){


	uint64_t num_teams = get_num_teams();

	attach_buffers(items, nitems);

	bulk_query_kernel<<<num_teams, BLOCK_SIZE>>>(this, items, hits);


}

//speed up queries by sorting to minimize memory movements required.
// This will sort the query list but won't mutate the items themselves
__host__ void optimized_vqf::sorted_bulk_query(uint64_t * items, uint64_t nitems, bool * hits){


	uint64_t num_teams = get_num_teams();

	attach_buffers(items, nitems);

	bulk_sorted_query_kernel<<<num_teams, BLOCK_SIZE>>>(this, items, hits);


}



__device__ bool optimized_vqf::mini_filter_bulk_queries(uint64_t * items, bool * hits){

	__shared__ thread_team_block block;

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t blockID = blockIdx.x;

	int warpID = threadIdx.x / 32;

	int threadID = threadIdx.x % 32;


	if (blockID >= num_teams) return false;



	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		block.internal_blocks[i] = blocks[blockID].internal_blocks[i]; 
		//printf("i: %d\n",i);
		
	}

	//separate these pulls so that the queries can be encapsulated

	__syncthreads();

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		//global buffer blockID*BLOCKS_PER_THREAD_BLOCK + i

		uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

		uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

		bool * hits_ptr = hits + global_offset;

		block.internal_blocks[i].sorted_bulk_query(threadID, buffers[global_buffer], hits_ptr, buffer_sizes[global_buffer]);

	}


	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

		uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

		bool * hits_ptr = hits + global_offset;


		for (int j = threadID; j < buffer_sizes[global_buffer]; j+=32){

			if (!hits_ptr[j]){

				uint64_t item = buffers[global_buffer][j];

				int alt_bucket = get_bucket_from_hash(get_alt_hash(item, global_buffer)) % BLOCKS_PER_THREAD_BLOCK;

				if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

				hits_ptr[j] = block.internal_blocks[alt_bucket].binary_search_query(item);
			}

		}
	}

	return true;

}

//passing along items so that global addresses into hits can be calculated.
__device__ bool optimized_vqf::mini_filter_queries(uint64_t * items, bool * hits){

	__shared__ thread_team_block block;

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t blockID = blockIdx.x;

	int warpID = threadIdx.x / 32;

	int threadID = threadIdx.x % 32;

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i]; 

		
	}

	//separate these pulls so that the queries can be encapsulated

	__syncthreads();


	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		query_single_buffer_block(&block, blockID, i, threadID, items, hits);
	}
}


__device__ bool optimized_vqf::mini_filter_block(uint64_t * misses){

	__shared__ thread_team_block block;


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t blockID = blockIdx.x;

	int warpID = threadIdx.x / 32;

	int threadID = threadIdx.x % 32;

	//each warp should grab one block
	//TODO modify for #filter blocks per thread_team_block

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

	block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];

	insert_single_buffer_block(&block, blockID, i, threadID);


	}


	__syncthreads();


	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

   	dump_remaining_buffers_block(&block, blockID, i, threadID, misses);

   	}

   __syncthreads();


  //  for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

  //   //relock blocks so that consistency assertions pass
  //  	block.internal_blocks[i].lock_local(threadID);
 
 	// }

   __syncthreads();

   	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

	blocks[blockIdx.x].internal_blocks[i] = block.internal_blocks[i];

	}

	return true;

}



__device__ bool optimized_vqf::sorted_mini_filter_block(uint64_t * misses){

	__shared__ thread_team_block block;


	#if TAG_BITS == 8

	__shared__ uint8_t temp_tags[BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK];


	#elif TAG_BITS == 16

	__shared__ uint16_t temp_tags[BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK];

	#endif


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t blockID = blockIdx.x;

	int warpID = threadIdx.x / 32;

	int threadID = threadIdx.x % 32;

	//each warp should grab one block
	//TODO modify for #filter blocks per thread_team_block

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


	//speed this up - should be .5-4 memory operations?
	//change this to look through uint64_T to load


	block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];


	#if TAG_BITS == 8

		sorted_insert_single_buffer_block(&block, (uint8_t *) &temp_tags, blockID, i, warpID, threadID);

	#elif TAG_BITS == 16

		sorted_insert_single_buffer_block(&block, (uint16_t *) &temp_tags, blockID, i, warpID, threadID);

	#endif

	//sorted_insert_single_buffer_block(&block, (uint8_t *) &temp_tags, blockID, i, warpID, threadID);


	}


	//return;


	__syncthreads();




	//TODO: reinstate this, preferably with more sorting


	//this loop needs to be moved internally for the sorted version

	#if TAG_BITS == 8

	  	dump_remaining_buffers_sorted(&block, (uint8_t *) &temp_tags, blockID, warpID, threadID, misses);

	#elif TAG_BITS == 16

  	 	dump_remaining_buffers_sorted(&block, (uint16_t *) &temp_tags, blockID, warpID, threadID, misses);

	#endif




   __syncthreads();


  //  for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

  //   //relock blocks so that consistency assertions pass
  //  	block.internal_blocks[i].lock_local(threadID);
 
 	// }

   __syncthreads();

  for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

	blocks[blockIdx.x].internal_blocks[i] = block.internal_blocks[i];

	}



}



//avoid extra temp tags via writing directly to the block?
__device__ bool optimized_vqf::sorted_mini_filter_block_async_write(uint64_t * misses){

	__shared__ thread_team_block block;


	#if TAG_BITS == 8

	//__shared__ uint8_t temp_tags[BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK];


	#elif TAG_BITS == 16

	//__shared__ uint16_t temp_tags[BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK];




	#endif


	//counters required
	//global offset
	//#elements dumped in round 1
	//fill within block
	//length from fill
	__shared__ int offsets[BLOCKS_PER_THREAD_BLOCK];


	__shared__ int counters[BLOCKS_PER_THREAD_BLOCK];


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t blockID = blockIdx.x;

	int warpID = threadIdx.x / 32;

	int threadID = threadIdx.x % 32;

	//each warp should grab one block
	//TODO modify for #filter blocks per thread_team_block

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


	//speed this up - should be .5-4 memory operations?
	//change this to look through uint64_T to load


	//funky load

	//load size of 4 bytes - 128 / 32

	#ifndef TOTAL_INTS
	#define TOTAL_INTS BYTES_PER_CACHE_LINE * CACHE_LINES_PER_BLOCK / 4
	#endif


	for (int j = threadID; j < TOTAL_INTS; j+=32){

		((int *)(&block.internal_blocks[i]))[j] = ((int *)(&blocks[blockIdx.x].internal_blocks[i]))[j];

	}

	//block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];




	buffer_get_primary_count(&block, (int *) &offsets[0], blockID, i, warpID, threadID);



	//sorted_insert_single_buffer_block(&block, (uint8_t *) &temp_tags, blockID, i, warpID, threadID);


	}


	//return;


	__syncthreads();


	//at this point the block is loaded and offsets contain the pieces of global buffers to be dumped
	//dump remaining buffers seems to need a temp array :/
	//I don't think theres a way to get around that unfortunately
	//it wastes space at the end of every buffer passed in but it can't be avoided.


	//



	//TODO: reinstate this, preferably with more sorting


	//this loop needs to be moved internally for the sorted version

	// #if TAG_BITS == 8

	//   	dump_remaining_buffers_sorted(&block, (uint8_t *) &temp_tags, blockID, warpID, threadID, misses);

	// #elif TAG_BITS == 16

 //  	 	dump_remaining_buffers_sorted(&block, (uint16_t *) &temp_tags, blockID, warpID, threadID, misses);

	// #endif


	dump_all_buffers_sorted(&block, &offsets[0], &counters[0], blockID, warpID, threadID, misses);


   __syncthreads();


  //  for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

  //   //relock blocks so that consistency assertions pass
  //  	block.internal_blocks[i].lock_local(threadID);
 
 	// }

  // __syncthreads();

 //  for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

	// blocks[blockIdx.x].internal_blocks[i] = block.internal_blocks[i];

	// }



}


__device__ bool optimized_vqf::query_single_buffer_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID, uint64_t * items, bool * hits){


	#if DEBUG_ASSERTS

	assert(blockID < num_teams);

	assert((blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);

	#endif

	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;

	#if DEBUG_ASSERTS

	assert(global_buffer / BLOCKS_PER_THREAD_BLOCK == blockID);
	assert(global_buffer % BLOCKS_PER_THREAD_BLOCK == warpID);

	//if this passes warpID == internalID

	#endif


	uint64_t global_offset = buffers[global_buffer] - items;

	int buf_size = buffer_sizes[global_buffer];

	for (int i = 0; i < buf_size; i++){


		bool found = false;


		//first query

		//uint64_t hash = hash_key(buffers[global_buffer][i]);

		uint64_t hash = buffers[global_buffer][i];


	   #if DEBUG_ASSERTS

		assert(get_bucket_from_hash(hash) == global_buffer);

	 	assert(local_blocks->internal_blocks[warpID].assert_consistency());

	    #endif

	   found = local_blocks->internal_blocks[warpID].query(threadID, hash);

	   #if DEBUG_ASSERTS
	   assert(local_blocks->internal_blocks[warpID].assert_consistency());
	   #endif


		

		//end of first query

		if (!found){

			 #if DEBUG_ASSERTS
	 		
	 		assert(local_blocks->internal_blocks[warpID].assert_consistency());


	 		#endif


	 		uint64_t alt_hash = get_alt_hash(hash, global_buffer);

	 		int alt_bucket = get_bucket_from_hash(alt_hash) % (BLOCKS_PER_THREAD_BLOCK);


	 		//does warpID == internalID
	 		//internalID = block_index % BLOCKS_PER_THREAD_BLOCK;
			if (alt_bucket == warpID) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;



			found = local_blocks->internal_blocks[alt_bucket].query(threadID, hash);

		}




		hits[global_offset+i] = found;
	}


}

//dump up to FILL_CUTOFF into each block
// this is handled at a per warp level, and this version does not rely on cooperative groups cause they slow down cudagdb
__device__ bool optimized_vqf::insert_single_buffer_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID){



	#if DEBUG_ASSERTS

	assert(blockID  < num_teams);

	assert( (blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);

	

	#endif

	//at this point the team should be referring to a valid target for insertions
	//this is a copy of buffer_insert modified to the use the cooperative group API
	//for the original version check optimized_vqf.cu::buffer_insert

	//local_blocks->internal_blocks[buffer];

	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;


	#if DEBUG_ASSERTS

	assert(byte_assert_sorted(buffers[global_buffer], buffer_sizes[global_buffer]));

	#endif



	//TODO: chance FILL_CUTOFF to be a percentage fill ratio
	int count = FILL_CUTOFF - local_blocks->internal_blocks[warpID].get_fill();

	int buf_size = buffer_sizes[global_buffer];

	if (buf_size < count) count = buf_size;


	// if (warpGroup.thread_rank() != 0){
	// 	printf("Halp: %d %d\n", warpGroup.thread_rank(), count);
	// }

	//modify to be warp group specific

	local_blocks->internal_blocks[warpID].bulk_insert(threadID, buffers[global_buffer], count);

	//local_blocks->internal_blocks[warpID].unlock(threadID);

	//block.bulk_insert(warpGroup.thread_rank(), buffers[global_buffer], count);


	//local_blocks->internal_blocks[buffer] = block;

	if (threadID == 0){

		buffers[global_buffer] += count;

		buffer_sizes[global_buffer] -= count;
	}

}


//dump up to FILL_CUTOFF into each block
// this is handled at a per warp level, and this version does not rely on cooperative groups cause they slow down cudagdb

#if TAG_BITS == 8
	
	__device__ bool optimized_vqf::sorted_insert_single_buffer_block(thread_team_block * local_blocks, uint8_t * temp_tags, uint64_t blockID, int warpID, int block_warpID, int threadID)

#elif TAG_BITS == 16

	__device__ bool optimized_vqf::sorted_insert_single_buffer_block(thread_team_block * local_blocks, uint16_t * temp_tags, uint64_t blockID, int warpID, int block_warpID, int threadID)

#endif

	{



	#if DEBUG_ASSERTS

	assert(blockID  < num_teams);

	assert( (blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);
	

	#endif  

	//at this point the team should be referring to a valid target for insertions
	//this is a copy of buffer_insert modified to the use the cooperative group API
	//for the original version check optimized_vqf.cu::buffer_insert

	//local_blocks->internal_blocks[buffer];

	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;


	#if DEBUG_ASSERTS



		#if TAG_BITS == 8

			assert(byte_assert_sorted(buffers[global_buffer], buffer_sizes[global_buffer]));

		#elif TAG_BITS == 16

			assert(two_byte_assert_sorted(buffers[global_buffer], buffer_sizes[global_buffer]));

		#endif

	

	#endif



	//TODO: chance FILL_CUTOFF to be a percentage fill ratio
	int count = FILL_CUTOFF - local_blocks->internal_blocks[warpID].get_fill();

	int buf_size = buffer_sizes[global_buffer];

	if (buf_size < count) count = buf_size;

	if (count < 0) count = 0;

	#if DEBUG_ASSERTS

	assert(count < SLOTS_PER_BLOCK);

	#endif

	// if (warpGroup.thread_rank() != 0){
	// 	printf("Halp: %d %d\n", warpGroup.thread_rank(), count);
	// }

	//modify to be warp group specific

	//local_blocks->internal_blocks[warpID].bulk_insert(threadID, buffers[global_buffer], count);

	//TODO: double check this is correct
	//I think this change is good but idk

	local_blocks->internal_blocks[warpID].sorted_bulk_insert(&temp_tags[warpID*SLOTS_PER_BLOCK], buffers[global_buffer], count, block_warpID, threadID);

	//local_blocks->internal_blocks[warpID].unlock(threadID);

	//block.bulk_insert(warpGroup.thread_rank(), buffers[global_buffer], count);


	//local_blocks->internal_blocks[buffer] = block;

	if (threadID == 0){

		buffers[global_buffer] += count;

		buffer_sizes[global_buffer] -= count;
	}

}


//dump up to FILL_CUTOFF into each block
// this is handled at a per warp level, and this version does not rely on cooperative groups cause they slow down cudagdb


__device__ bool optimized_vqf::buffer_get_primary_count(thread_team_block * local_blocks, int * counters, uint64_t blockID, int warpID, int block_warpID, int threadID){



	#if DEBUG_ASSERTS

	assert(blockID  < num_teams);

	assert( (blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);
	

	#endif  

	//at this point the team should be referring to a valid target for insertions
	//this is a copy of buffer_insert modified to the use the cooperative group API
	//for the original version check optimized_vqf.cu::buffer_insert

	//local_blocks->internal_blocks[buffer];

	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;


	#if DEBUG_ASSERTS



		#if TAG_BITS == 8

			assert(byte_assert_sorted(buffers[global_buffer], buffer_sizes[global_buffer]));

		#elif TAG_BITS == 16

			assert(two_byte_assert_sorted(buffers[global_buffer], buffer_sizes[global_buffer]));

		#endif

	

	#endif



	//TODO: chance FILL_CUTOFF to be a percentage fill ratio
	int count = FILL_CUTOFF - local_blocks->internal_blocks[warpID].get_fill();

	int buf_size = buffer_sizes[global_buffer];

	if (buf_size < count) count = buf_size;

	if (count < 0) count = 0;

	#if DEBUG_ASSERTS

	assert(count < SLOTS_PER_BLOCK);

	#endif


	//this thread only needs to prep this for later! - global buffer is sorted and ready.
	counters[warpID] = count;

	// if (warpGroup.thread_rank() != 0){
	// 	printf("Halp: %d %d\n", warpGroup.thread_rank(), count);
	// }

	//modify to be warp group specific

	//local_blocks->internal_blocks[warpID].bulk_insert(threadID, buffers[global_buffer], count);

	//TODO: double check this is correct
	//I think this change is good but idk

	// local_blocks->internal_blocks[warpID].sorted_bulk_insert(&temp_tags[warpID*SLOTS_PER_BLOCK], buffers[global_buffer], count, block_warpID, threadID);

	// //local_blocks->internal_blocks[warpID].unlock(threadID);

	// //block.bulk_insert(warpGroup.thread_rank(), buffers[global_buffer], count);


	// //local_blocks->internal_blocks[buffer] = block;

	// if (threadID == 0){

	// 	buffers[global_buffer] += count;

	// 	buffer_sizes[global_buffer] -= count;
	// }

}



//dump_remaining_buffers_sorted
//allocate space for a new zip algorithm and move items to the correct bucket
//we need at most 25% of the space?

#if TAG_BITS == 8

__device__ void optimized_vqf::dump_remaining_buffers_sorted(thread_team_block * local_blocks, uint8_t * temp_tags, uint64_t blockID, int warpID, int threadID, uint64_t * misses)


#elif TAG_BITS == 16

__device__ void optimized_vqf::dump_remaining_buffers_sorted(thread_team_block * local_blocks, uint16_t * temp_tags, uint64_t blockID, int warpID, int threadID, uint64_t * misses)


#endif



{


	//get remaining keys

	//how much shared mem can we work with - if this allocation fails use the same buffers as before?

	__shared__ int counters [BLOCKS_PER_THREAD_BLOCK];

	__shared__ int start_counters[BLOCKS_PER_THREAD_BLOCK];


	__syncthreads();


	//evenly partition
	if (threadID == 0){

		for (int i =warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


		counters[i] = local_blocks->internal_blocks[i].get_fill();

		start_counters[i] = 0;

		}

	}


	__syncthreads();


	#if DEBUG_ASSERTS


	for(int i = 0; i < BLOCKS_PER_THREAD_BLOCK; i++){
		assert(start_counters[i] == 0);
	}

	__syncthreads();
	#endif


	//TINY OPTIMIZATION
	// the two add operations aren't necessary, the primary record should be sufficient
	// change once I get this working.

	int slot;

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


		//for each item in parallel, we check the global counters to determine which hash is submitted
		uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;

		int remaining = buffer_sizes[global_buffer];

		for (int j = threadID; j < remaining; j+=32){


			uint64_t hash = buffers[global_buffer][j];

			uint64_t alt_hash = get_alt_hash(hash, global_buffer);

			int alt_bucket = get_bucket_from_hash(alt_hash) % BLOCKS_PER_THREAD_BLOCK;

			if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


			#if DEBUG_ASSERTS

			assert(alt_bucket < BLOCKS_PER_THREAD_BLOCK);
			assert(i < BLOCKS_PER_THREAD_BLOCK);


			#endif

			//replace with faster atomic
			if 	(atomicCAS(&counters[i], (int) 0, (int) 0) < atomicCAS(&counters[alt_bucket], (int) 0, (int) 0)){


				slot = atomicAdd(&counters[i], 1);

				//These adds aren't undone on failure as no one else can succeed.
				if (slot < SLOTS_PER_BLOCK){

					slot = atomicAdd(&start_counters[i],1);

					#if TAG_BITS == 8

						temp_tags[i*SLOTS_PER_BLOCK+slot] = hash & 0xff;

					#elif TAG_BITS == 16

						temp_tags[i*SLOTS_PER_BLOCK+slot] = hash & 0xffff;

					#endif

				


					#if DEBUG_ASSERTS

					assert(slot + local_blocks->internal_blocks[i].get_fill() < SLOTS_PER_BLOCK);

					#endif

				} else {

					//atomicadd fails, try alternate spot
					slot = atomicAdd(&counters[alt_bucket], 1);

					if (slot < SLOTS_PER_BLOCK){

						slot = atomicAdd(&start_counters[alt_bucket], 1);

						

						#if TAG_BITS == 8

							temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xff;	

						#elif TAG_BITS == 16

							temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xffff;	

						#endif

						#if DEBUG_ASSERTS

						assert(slot + local_blocks->internal_blocks[alt_bucket].get_fill() < SLOTS_PER_BLOCK);

						#endif					

					} else {


						atomicAdd((unsigned long long int *) misses, 1ULL);

					}



				}


			} else {

				//alt < main slot
				slot = atomicAdd(&counters[alt_bucket], 1);

				if (slot < SLOTS_PER_BLOCK){

					slot = atomicAdd(&start_counters[alt_bucket], 1);

					//temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xff;

					#if TAG_BITS == 8

						temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xff;	

					#elif TAG_BITS == 16

						temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xffff;	

					#endif

					#if DEBUG_ASSERTS

					assert(slot + local_blocks->internal_blocks[alt_bucket].get_fill() < SLOTS_PER_BLOCK);

					#endif

				} else {


					//primary insert failed, attempt secondary
					slot = atomicAdd(&counters[i], 1);

					if (slot < SLOTS_PER_BLOCK){

						slot = atomicAdd(&start_counters[i],1);

						//temp_tags[i*SLOTS_PER_BLOCK+slot] = hash & 0xff;

						#if TAG_BITS == 8

							temp_tags[i*SLOTS_PER_BLOCK+slot] = hash & 0xff;

						#elif TAG_BITS == 16

							temp_tags[i*SLOTS_PER_BLOCK+slot] = hash & 0xffff;

						#endif

						#if DEBUG_ASSERTS

						assert(slot + local_blocks->internal_blocks[i].get_fill() < SLOTS_PER_BLOCK);

						#endif

					} else {



						atomicAdd((unsigned long long int *) misses, 1ULL);


						}

					}




			}


		}

	}



	__syncthreads();



	//sort and dump

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		int length = start_counters[i];


		#if DEBUG_ASSERTS

		if (! (start_counters[i] + local_blocks->internal_blocks[i].get_fill() <= SLOTS_PER_BLOCK)){

				//start_counters[i] -1
				assert(start_counters[i] + local_blocks->internal_blocks[i].get_fill() <= SLOTS_PER_BLOCK);

		}

	

		assert(length + local_blocks->internal_blocks[i].get_fill() <= SLOTS_PER_BLOCK);

		#endif



		// if (length > 32 && threadID == 0)

		// 		insertion_sort_max(&temp_tags[i*SLOTS_PER_BLOCK], length);

		// 	sorting_network_8_bit(&temp_tags[i*SLOTS_PER_BLOCK], length, threadID);

		// 	__syncwarp();

		// 	#if DEBUG_ASSERTS

		// 	assert(short_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], length));

		// 	#endif


		//EOD HERE - patch sorting network for 16 bit 


		#if TAG_BITS == 8


		//start of 8 bit

		if (length <= 32){


			sorting_network_8_bit(&temp_tags[i*SLOTS_PER_BLOCK], length, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS

			assert(short_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], length));

			#endif

		}	else {


			// if (threadID ==0)

			// insertion_sort_max(&temp_tags[i*SLOTS_PER_BLOCK], length);

	

			__syncwarp();

			sorting_network_8_bit(&temp_tags[i*SLOTS_PER_BLOCK], 32, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS


			assert(short_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], 32));

			assert(short_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], length));

			#endif

		}

		//end of 8 bit


		#elif TAG_BITS == 16

		//start of 16 bit

		if (length <= 32){


			sorting_network_16_bit(&temp_tags[i*SLOTS_PER_BLOCK], length, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS

			assert(sixteen_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], length));

			#endif

		}	else {


			if (threadID ==0)

			insertion_sort_16(&temp_tags[i*SLOTS_PER_BLOCK], length);

	

			__syncwarp();

			sorting_network_16_bit(&temp_tags[i*SLOTS_PER_BLOCK], 32, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS


			assert(sixteen_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], 32));

			assert(sixteen_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], length));

			#endif

		}

		//end of 16 bit


		#endif




		local_blocks->internal_blocks[i].sorted_bulk_finish(&temp_tags[i*SLOTS_PER_BLOCK+length], &temp_tags[i*SLOTS_PER_BLOCK], length, warpID, threadID);



		//and merge into main arrays



	}

	// #if DEBUG_ASSERTS

	// assert(blockID < num_teams);

	// assert((blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);

	// #endif

	// uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;

	// int remaining = buffer_sizes[global_buffer];



	// for (int i = threadID; i < remaining; i+=32){

	// 	uint64_t hash = buffers[global_buffer][i];


	// 	#if DEBUG_ASSERTS

	// 	assert(get_bucket_from_hash(hash) == global_buffer);


	// 	#endif

	// 	uint64_t alt_hash = get_alt_hash(hash, global_buffer);

	// 	int alt_bucket = get_bucket_from_hash(alt_hash) % (BLOCKS_PER_THREAD_BLOCK);

	// 	if (alt_bucket == warpID) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;

	// 	//copied over from lock_blocks, but for the local case (use shared mem atomics)


	// 	//blocks are locked

	// 	if (local_blocks->internal_blocks[warpID].get_fill_atomic() < local_blocks->internal_blocks[alt_bucket].get_fill_atomic()){

	// 		//TODO: verify queries also check for just hash
	// 		if (!local_blocks->internal_blocks[warpID].insert_one_thread(hash)){


	// 			if (!local_blocks->internal_blocks[alt_bucket].insert_one_thread(hash)){

	// 				atomicAdd((unsigned long long int *) misses, 1);

	// 			}
				

	// 		}
	// 	} else {
	// 		if(!local_blocks->internal_blocks[alt_bucket].insert_one_thread(hash)){


	// 			if (!local_blocks->internal_blocks[warpID].insert_one_thread(hash)){

	// 				atomicAdd((unsigned long long int *) misses, 1);
					
	// 			}

				

	// 		}
	// 	}

	// //end of main loop
	// }

}



//dump_remaining_buffers_sorted
//allocate space for a new zip algorithm and move items to the correct bucket
//we need at most 25% of the space?


__device__ bool optimized_vqf::query_single_item_sorted_debug(int warpID, uint64_t hash){


	//pull info from item

	uint64_t global_bucket = get_bucket_from_hash(hash);

	uint64_t blockID = global_bucket / BLOCKS_PER_THREAD_BLOCK;

	int internalID = global_bucket % BLOCKS_PER_THREAD_BLOCK;


	bool found = blocks[blockID].internal_blocks[internalID].query(warpID, hash);

	if (found) return true;

	uint64_t alt_hash = get_alt_hash(hash, global_bucket);

	int alt_bucket = get_bucket_from_hash(alt_hash) % BLOCKS_PER_THREAD_BLOCK;

	if (alt_bucket == internalID) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


	return blocks[blockID].internal_blocks[alt_bucket].query(warpID, hash);

}



__device__ void optimized_vqf::dump_all_buffers_sorted(thread_team_block * local_blocks, int * offsets, int * counters, uint64_t blockID, int warpID, int threadID, uint64_t * misses)



{


	//get remaining keys

	//how much shared mem can we work with - if this allocation fails use the same buffers as before?

	// __shared__ int counters [BLOCKS_PER_THREAD_BLOCK];

	// __shared__ int start_counters[BLOCKS_PER_THREAD_BLOCK];


	__syncthreads();



	//evenly partition
	if (threadID == 0){

		for (int i =warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


		//remaining counters now takes into account the main list as well as new inserts

		counters[i] = offsets[i] + local_blocks->internal_blocks[i].get_fill();

		//start_counters[i] = 0;

		}

	}


	__syncthreads();


	#if DEBUG_ASSERTS


	// for(int i = 0; i < BLOCKS_PER_THREAD_BLOCK; i++){
	// 	assert(start_counters[i] == 0);
	// }

	for (int i =0; i < BLOCKS_PER_THREAD_BLOCK; i++){
		assert(counters[i] <= SLOTS_PER_BLOCK);
	}

	__syncthreads();
	#endif


	//TINY OPTIMIZATION
	// the two add operations aren't necessary, the primary record should be sufficient
	// change once I get this working.

	int slot;

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


		//for each item in parallel, we check the global counters to determine which hash is submitted
		uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;

		int remaining = buffer_sizes[global_buffer] - offsets[i];

		for (int j = threadID; j < remaining; j+=32){


			uint64_t hash = buffers[global_buffer][j+offsets[i]];

			uint64_t alt_hash = get_alt_hash(hash, global_buffer);

			int alt_bucket = get_bucket_from_hash(alt_hash) % BLOCKS_PER_THREAD_BLOCK;

			if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


			#if DEBUG_ASSERTS

			assert(j + offsets[i] < buffer_sizes[global_buffer]);

			assert(alt_bucket < BLOCKS_PER_THREAD_BLOCK);
			assert(i < BLOCKS_PER_THREAD_BLOCK);
			assert(alt_bucket != i);


			#endif

			//replace with faster atomic

			//
			//if 	(atomicAdd(&counters[i], (int) 0) < atomicAdd(&counters[alt_bucket], (int) 0)){
			if (atomicCAS(&counters[i], 0, 0) < atomicCAS(&counters[alt_bucket],0,0)){

				slot = atomicAdd(&counters[i], 1);

				//These adds aren't undone on failure as no one else can succeed.
				if (slot < SLOTS_PER_BLOCK){

					//slot - offset = fill+#writes - this is guaranteed to be a free slot
					slot -= offsets[i];

					#if TAG_BITS == 8

						local_blocks->internal_blocks[i].tags[slot] = hash & 0xff;

					#elif TAG_BITS == 16

						local_blocks->internal_blocks[i].tags[slot] = hash & 0xffff;

					#endif

				


					#if DEBUG_ASSERTS

					assert(slot + offsets[i]  < SLOTS_PER_BLOCK);

					#endif

				} else {

					//atomicSub(&counters[i],1);

					//atomicadd fails, try alternate spot
					slot = atomicAdd(&counters[alt_bucket], 1);

					if (slot < SLOTS_PER_BLOCK){

						slot -= offsets[alt_bucket];

						

						#if TAG_BITS == 8

							//temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xff;	
							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash & 0xff;

						#elif TAG_BITS == 16

							//temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xffff;	
							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash & 0xffff;

						#endif

						#if DEBUG_ASSERTS

						assert(slot + offsets[alt_bucket] < SLOTS_PER_BLOCK);

						#endif					

					} else {

						//atomicSub(&counters[alt_bucket],1);

						atomicAdd((unsigned long long int *) misses, 1ULL);

					}



				}


			} else {

				//alt < main slot
				slot = atomicAdd(&counters[alt_bucket], 1);

				if (slot < SLOTS_PER_BLOCK){

					//slot = atomicAdd(&start_counters[alt_bucket], 1);
					slot -= offsets[alt_bucket];

					//temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xff;

						#if TAG_BITS == 8

							//temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xff;	
							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash & 0xff;

						#elif TAG_BITS == 16

							//temp_tags[alt_bucket*SLOTS_PER_BLOCK+slot] = hash & 0xffff;	
							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash & 0xffff;

						#endif

						#if DEBUG_ASSERTS

						assert(slot + offsets[alt_bucket] < SLOTS_PER_BLOCK);

						#endif		

				} else {

					//atomicSub(&counters[alt_bucket], 1); 

					//primary insert failed, attempt secondary
					slot = atomicAdd(&counters[i], 1);

					if (slot < SLOTS_PER_BLOCK){

						slot -= offsets[i];

						//temp_tags[i*SLOTS_PER_BLOCK+slot] = hash & 0xff;

						#if TAG_BITS == 8

							local_blocks->internal_blocks[i].tags[slot] = hash & 0xff;

						#elif TAG_BITS == 16

							local_blocks->internal_blocks[i].tags[slot] = hash & 0xffff;

						#endif

					


						#if DEBUG_ASSERTS

						assert(slot + offsets[i]  < SLOTS_PER_BLOCK);

						#endif

					} else {


						//atomicSub(&counters[alt_bucket], 1);
						atomicAdd((unsigned long long int *) misses, 1ULL);


						}

					}




			}


		}

	}



	__syncthreads();


	#if DEBUG_ASSERTS


	//loop and look for items in temp buffers
	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		//we want to assert that every item not in our main buffer made it to one of its temp

		//we will check before and after sorting

		if (counters[i] > SLOTS_PER_BLOCK){

			counters[i] = SLOTS_PER_BLOCK;

		}


	}

	//assert(misses[0] != 0);

	__syncthreads();

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;

		for (int j = offsets[i]; j < buffer_sizes[global_buffer]; j++){

			bool found = false;

			//check if first contains
			//main buffer runs from temp_tags, starts at get_fill(), runs to length

			uint64_t hash = buffers[global_buffer][j];

			uint64_t alt_hash = get_alt_hash(hash, global_buffer);

			int alt_bucket = get_bucket_from_hash(alt_hash) % BLOCKS_PER_THREAD_BLOCK;

			if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;

			int check_length = counters[i] - offsets[i] - local_blocks->internal_blocks[i].get_fill();

			for (int k = local_blocks->internal_blocks[i].get_fill(); k < check_length; k++){

				if (local_blocks->internal_blocks[i].tags[k] == (hash & 0xffff)) found = true;
			}



			//now check alt_bucket

			int alt_length = counters[alt_bucket]- offsets[alt_bucket] - local_blocks->internal_blocks[alt_bucket].get_fill();

			for (int k = local_blocks->internal_blocks[alt_bucket].get_fill(); k < alt_length; k++){

				if (local_blocks->internal_blocks[alt_bucket].tags[k] == (hash & 0xffff)) found = true;
			}


			if (!found){
				//assert(found);
				continue;
			}
			

		}


	}

	__syncthreads();

	#endif


	//at this point global buffer is ready, and so are the temp buffers (after sorting)



	//sort and dump

	for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		if (counters[i] > SLOTS_PER_BLOCK){

			counters[i] = SLOTS_PER_BLOCK;

		}

		#if DEBUG_ASSERTS

		if (counters[i] > SLOTS_PER_BLOCK){

			counters[i] = SLOTS_PER_BLOCK;

			assert(counters[i] <= SLOTS_PER_BLOCK);
		}
		

		#endif


		//
		int length = counters[i] - offsets[i] - local_blocks->internal_blocks[i].get_fill();
 


		#if DEBUG_ASSERTS

		if (length + local_blocks->internal_blocks[i].get_fill() + offsets[i] > SLOTS_PER_BLOCK){

			assert(length + local_blocks->internal_blocks[i].get_fill() + offsets[i] <= SLOTS_PER_BLOCK);

		}
	

		if (! (counters[i] <= SLOTS_PER_BLOCK)){

				//start_counters[i] -1
				assert(counters[i] <= SLOTS_PER_BLOCK);

		}

	

		#endif



		// if (length > 32 && threadID == 0)

		// 		insertion_sort_max(&temp_tags[i*SLOTS_PER_BLOCK], length);

		// 	sorting_network_8_bit(&temp_tags[i*SLOTS_PER_BLOCK], length, threadID);

		// 	__syncwarp();

		// 	#if DEBUG_ASSERTS

		// 	assert(short_byte_assert_sorted(&temp_tags[i*SLOTS_PER_BLOCK], length));

		// 	#endif


		//EOD HERE - patch sorting network for 16 bit 


		int tag_fill = local_blocks->internal_blocks[i].get_fill();

		#if TAG_BITS == 8


		//start of 8 bit

		if (length <= 32){

			#if DEBUG_ASSERTS

				assert(tag_fill + length <=SLOTS_PER_BLOCK);

			#endif
			


			sorting_network_8_bit(&local_blocks->internal_blocks[i].tags[tag_fill], length, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS

			assert(short_byte_assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			#endif

		}	else {


			if (threadID ==0)

			insertion_sort_max(&local_blocks->internal_blocks[i].tags[tag_fill], length);

	

			__syncwarp();

			sorting_network_8_bit(&local_blocks->internal_blocks[i].tags[tag_fill], 32, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS


			assert(short_byte_assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], 32));

			assert(short_byte_assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			#endif

		}

		//end of 8 bit


		#elif TAG_BITS == 16

		//start of 16 bit

		if (length <= 32){

			#if DEBUG_ASSERTS

				assert(tag_fill + length <=SLOTS_PER_BLOCK);

			#endif


			sorting_network_16_bit(&local_blocks->internal_blocks[i].tags[tag_fill], length, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS

			assert(sixteen_byte_assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			#endif

		}	else {


			#if DEBUG_ASSERTS

				assert(tag_fill + length <=SLOTS_PER_BLOCK);

			#endif


			if (threadID ==0)

			insertion_sort_16(&local_blocks->internal_blocks[i].tags[tag_fill], length);

	

			__syncwarp();

			sorting_network_16_bit(&local_blocks->internal_blocks[i].tags[tag_fill], 32, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS


			assert(sixteen_byte_assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], 32));

			assert(sixteen_byte_assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			#endif

		}

		//end of 16 bit

		assert(length + tag_fill + offsets[i] <= SLOTS_PER_BLOCK);


		#endif


		//now all three arrays are sorted, and we have a valid target for write-out



		//local_blocks->internal_blocks[i].sorted_bulk_finish(&temp_tags[i*SLOTS_PER_BLOCK+length], &temp_tags[i*SLOTS_PER_BLOCK], length, warpID, threadID);




		//and merge into main arrays
		uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


		//buffers to be dumped
		//global_buffer -> counter starts at 0, runs to offets[i];
		//temp_tags, starts at 0, runs to get_fill();
		//other temp_tags, starts at get_fill(), runs to length; :D


		blocks[blockID].internal_blocks[i].dump_all_buffers_sorted(buffers[global_buffer], offsets[i], &local_blocks->internal_blocks[i].tags[0], tag_fill, &local_blocks->internal_blocks[i].tags[tag_fill], length, warpID, threadID);

		//double triple check that dump_all_buffers increments the internal counts like it needs to.


		//maybe this is the magic?

		#if DEBUG_ASSERTS
		__threadfence();


		if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) != offsets[i]){

			assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) == offsets[i]);

		}

		if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) != tag_fill){

			assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) == tag_fill);
	

		}


		if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) != length ){

			assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) == length);

		}


		//assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], buffer_sizes[global_buffer]) == buffer_sizes[global_buffer]);

		
		#endif

	}



	#if DEBUG_ASSERTS


	__threadfence();

	//let everyone do all checks
	// for (int i =0; i < BLOCKS_PER_THREAD_BLOCK; i+=1){

	// 	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


	// 	for (int j = 0; j < buffer_sizes[global_buffer]; j++){

	// 		assert(query_single_item_sorted_debug(threadID, buffers[global_buffer][j]));
	// 	}

	// }

	#endif



	}

// //dump up to FILL_CUTOFF into each block
// // this is handled at a per warp level, and this version does not rely on cooperative groups cause they slow down cudagdb
// __device__ bool optimized_vqf::sorted_insert_single_buffer_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int block_warpID, int threadID){



// 	#if DEBUG_ASSERTS

// 	assert(blockID  < num_teams);

// 	assert( (blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);
	

// 	#endif  

// 	//at this point the team should be referring to a valid target for insertions
// 	//this is a copy of buffer_insert modified to the use the cooperative group API
// 	//for the original version check optimized_vqf.cu::buffer_insert

// 	//local_blocks->internal_blocks[buffer];

// 	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;


// 	#if DEBUG_ASSERTS

// 	assert(byte_assert_sorted(buffers[global_buffer], buffer_sizes[global_buffer]));

// 	#endif



// 	//TODO: chance FILL_CUTOFF to be a percentage fill ratio
// 	int count = FILL_CUTOFF - local_blocks->internal_blocks[warpID].get_fill();

// 	int buf_size = buffer_sizes[global_buffer];

// 	if (buf_size < count) count = buf_size;


// 	// if (warpGroup.thread_rank() != 0){
// 	// 	printf("Halp: %d %d\n", warpGroup.thread_rank(), count);
// 	// }

// 	//modify to be warp group specific

// 	//local_blocks->internal_blocks[warpID].bulk_insert(threadID, buffers[global_buffer], count);

// 	local_blocks->internal_blocks[warpID].sorted_bulk_insert(buffers[global_buffer], count, block_warpID, threadID);

// 	//local_blocks->internal_blocks[warpID].unlock(threadID);

// 	//block.bulk_insert(warpGroup.thread_rank(), buffers[global_buffer], count);


// 	//local_blocks->internal_blocks[buffer] = block;

// 	if (threadID == 0){

// 		buffers[global_buffer] += count;

// 		buffer_sizes[global_buffer] -= count;
// 	}

// }


__device__ void optimized_vqf::dump_remaining_buffers_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID, uint64_t * misses){


	//get remaining keys


	#if DEBUG_ASSERTS

	assert(blockID < num_teams);

	assert((blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);

	#endif

	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;

	int remaining = buffer_sizes[global_buffer];



	for (int i = threadID; i < remaining; i+=32){

		uint64_t hash = buffers[global_buffer][i];


		#if DEBUG_ASSERTS

		assert(get_bucket_from_hash(hash) == global_buffer);


		#endif

		uint64_t alt_hash = get_alt_hash(hash, global_buffer);

		int alt_bucket = get_bucket_from_hash(alt_hash) % (BLOCKS_PER_THREAD_BLOCK);

		if (alt_bucket == warpID) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;

		//copied over from lock_blocks, but for the local case (use shared mem atomics)


		//blocks are locked

		if (local_blocks->internal_blocks[warpID].get_fill_atomic() < local_blocks->internal_blocks[alt_bucket].get_fill_atomic()){

			//TODO: verify queries also check for just hash
			if (!local_blocks->internal_blocks[warpID].insert_one_thread(hash)){


				if (!local_blocks->internal_blocks[alt_bucket].insert_one_thread(hash)){

					atomicAdd((unsigned long long int *) misses, 1);

				}
				

			}
		} else {
			if(!local_blocks->internal_blocks[alt_bucket].insert_one_thread(hash)){


				if (!local_blocks->internal_blocks[warpID].insert_one_thread(hash)){

					atomicAdd((unsigned long long int *) misses, 1);
					
				}

				

			}
		}

	//end of main loop
	}


	//TODO:




	// if (threadID == 0){
	// 	if (remaining >= 32){
	// 		printf("Case not handled atm\n");

	// 		atomicAdd((unsigned long long int *) misses, remaining-32);
	// 	} 
	// }
	
	

	// if (threadID < remaining){

	// 	uint64_t hash = buffers[global_buffer][threadID];


	// 	#if DEBUG_ASSERTS

	// 	assert(get_bucket_from_hash(hash) == global_buffer);


	// 	#endif

	// 	uint64_t alt_hash = get_alt_hash(hash, global_buffer);

	// 	int alt_bucket = get_bucket_from_hash(alt_hash) % (BLOCKS_PER_THREAD_BLOCK);

	// 	if (alt_bucket == warpID) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;

	// 	//copied over from lock_blocks, but for the local case (use shared mem atomics)


	// 	//blocks are locked

	// 	if (local_blocks->internal_blocks[warpID].get_fill_atomic() < local_blocks->internal_blocks[alt_bucket].get_fill_atomic()){

	// 		//TODO: verify queries also check for just hash
	// 		if (!local_blocks->internal_blocks[warpID].insert_one_thread(hash)){


	// 			if (!local_blocks->internal_blocks[alt_bucket].insert_one_thread(hash)){

	// 				atomicAdd((unsigned long long int *) misses, 1);

	// 			}
				

	// 		}
	// 	} else {
	// 		if(!local_blocks->internal_blocks[alt_bucket].insert_one_thread(hash)){


	// 			if (!local_blocks->internal_blocks[warpID].insert_one_thread(hash)){

	// 				atomicAdd((unsigned long long int *) misses, 1);

	// 			}

				

	// 		}
	// 	}

	// }

	

}
 



// __device__ bool optimized_vqf::finalize_thread_group(thread_group teamGroup, uint64_t teamID){



// 	//starting with teamGroup and threadGroup - this is just gonna go after the main code


// }


// __device__ void optimized_vqf::dump_thread_group_reserved(thread_group teamGroup, uint64_t teamID){


// 	//thread group local atomic? that would be great


// }


// __device__ int optimized_vqf::buffer_query(int warpID, uint64_t buffer){


// 	#if DEBUG_ASSERTS

// 	assert(buffer < num_blocks);

// 	#endif


// 	uint64_t block_index = buffer;

// 	lock_block(warpID, block_index);

	
// 	int buf_size = buffer_sizes[buffer];


// 	int found = blocks[block_index].bulk_query(warpID, buffers[buffer], buf_size);
	

// 	unlock_block(warpID, block_index);

// 	//and decrement the count



// 	return buf_size - found;


// }




// __device__ bool vqf::shared_buffer_insert(int warpID, int shared_blockID, uint64_t buffer){


// 	__shared__ vqf_block extern_blocks[WARPS_PER_BLOCK];


// 	#if DEBUG_ASSERTS

// 	assert(buffer < num_blocks);

// 	#endif


// 	uint64_t block_index = buffer;

// 	//lock_block(warpID, block_index);


// 	if (warpID == 0) extern_blocks[shared_blockID] = blocks[block_index];

// 	//extern_blocks[shared_blockID].load_block(warpID, blocks + block_index);

// 	extern_blocks[shared_blockID].lock(warpID);


// 	int fill_main = extern_blocks[shared_blockID].get_fill();

// 	#ifdef DEBUG_ASSERTS
// 	assert(fill_main == 0);
// 	#endif

// 	int count = FILL_CUTOFF - fill_main;

// 	int buf_size = buffer_sizes[buffer];

// 	if (buf_size < count) count = buf_size;

// 	for (int i =0; i < count; i++){

// 		#if DEBUG_ASSERTS

// 		int old_fill = extern_blocks[shared_blockID].get_fill();


// 		//relevant equation

// 		// (x mod yz) | z == x mod y?
// 		//python says no ur a dumbass this is the bug
		
// 		if (!(get_bucket_from_hash(buffers[buffer][i])  == buffer)){

// 			if (warpID == 0){

// 				printf("i %d count %d item %llu buffer %llu new_buf %llu\n", i, count, buffers[buffer][i], buffer, get_bucket_from_hash(buffers[buffer][i]));
// 			}

// 			__syncwarp();

// 			assert((buffers[buffer][i] >> TAG_BITS) % num_blocks  == buffer);

// 		}
		


// 		#endif

// 		uint64_t tag = buffers[buffer][i] & ((1ULL << TAG_BITS) -1);
// 		extern_blocks[shared_blockID].insert(warpID, tag);

// 		#if DEBUG_ASSERTS

// 		assert(extern_blocks[shared_blockID].get_fill() == old_fill+1);

// 		#endif

// 	}

// 	//write back

// 	extern_blocks[shared_blockID].unlock(warpID);


// 	//if (warpID == 0)
// 	//blocks[block_index] = extern_blocks[shared_blockID];

// 	//blocks[block_index].load_block(warpID, extern_blocks + shared_blockID);


// 	__threadfence();
// 	__syncwarp();

// 	//blocks[block_index].unlock(warpID);
	

// 	//and decrement the count

// 	if (warpID == 0){

// 		buffers[buffer] += count;

// 		buffer_sizes[buffer] -= count;


// 	}


// }


// __device__ bool vqf::multi_buffer_insert(int warpID, int init_blockID, uint64_t start_buffer){


// 	__shared__ vqf_block extern_blocks[WARPS_PER_BLOCK*REGIONS_PER_WARP];


// 	#if DEBUG_ASSERTS

// 	assert(start_buffer < num_blocks);

// 	#endif


// 	int shared_blockID = init_blockID * REGIONS_PER_WARP;



// 	if (start_buffer + warpID < num_blocks)

// 	{


// 		extern_blocks[shared_blockID + warpID % REGIONS_PER_WARP] = blocks[start_buffer + warpID % REGIONS_PER_WARP];

// 	}

// 	__syncwarp();

// 	for (int i = 0; i < REGIONS_PER_WARP; i++){

// 		if (start_buffer + i >= num_blocks) break;

// 		extern_blocks[shared_blockID + i].lock(warpID);
// 	}


// 	// 	


// 	// }

// 	__syncwarp();
	

// 	for (int i = 0; i < REGIONS_PER_WARP; i++){

// 		if (start_buffer + i >= num_blocks) break;

// 		int extern_id = shared_blockID + i;

// 		uint64_t buffer = start_buffer + i;



// 		int fill_main = extern_blocks[extern_id].get_fill();

// 		#ifdef DEBUG_ASSERTS
// 		assert(fill_main == 0);
// 		#endif

// 		int count = FILL_CUTOFF - fill_main;

// 		int buf_size = buffer_sizes[buffer];

// 		if (buf_size < count) count = buf_size;

// 		for (int i =0; i < count; i++){




// 			#if DEBUG_ASSERTS

// 			int old_fill = extern_blocks[extern_id].get_fill();


// 			//relevant equation

// 			// (x mod yz) | z == x mod y?
// 			//python says no ur a dumbass this is the bug
			
// 			if (!(get_bucket_from_hash(buffers[buffer][i])  == buffer)){

// 				if (warpID == 0){

// 					printf("i %d count %d item %llu buffer %llu new_buf %llu\n", i, count, buffers[buffer][i], buffer, get_bucket_from_hash(buffers[buffer][i]));
// 				}

// 				__syncwarp();

// 				assert((buffers[buffer][i] >> TAG_BITS) % num_blocks  == buffer);

// 			}
			


// 			#endif

// 			uint64_t tag = buffers[buffer][i] & ((1ULL << TAG_BITS) -1);
// 			extern_blocks[extern_id].insert(warpID, tag);

// 			#if DEBUG_ASSERTS

// 			assert(extern_blocks[extern_id].get_fill() == old_fill+1);

// 			#endif

// 		}


// 	//wrap up the loops

// 	extern_blocks[extern_id].unlock(warpID);

// 	if (warpID == 0){

// 		buffers[buffer] += count;

// 		buffer_sizes[buffer] -= count;


// 	}

// 	__syncwarp();

// 	}




// 	//write back

// 	for (int i = 0; i < REGIONS_PER_WARP; i++){

// 		if (start_buffer + i >= num_blocks) break;

		
// 		extern_blocks[shared_blockID + i].unlock(warpID);
// 	}


// 		if (start_buffer + warpID < num_blocks)

// 			{


// 			blocks[start_buffer + warpID % REGIONS_PER_WARP] = extern_blocks[shared_blockID + warpID % REGIONS_PER_WARP];


// 		}
// 	//if (warpID == 0)
// 	//blocks[block_index] = extern_blocks[shared_blockID];

// 	//blocks[block_index].load_block(warpID, extern_blocks + shared_blockID);


// 	__threadfence();
// 	__syncwarp();

// 	//blocks[block_index].unlock(warpID);
	

// 	//and decrement the count



// }

//Double check that the two inserts line up!
//to activate, tab out the code that changes the sizes of the buffers in buffer_insert
//otherwise results get wacky
// __device__ bool vqf::shared_buffer_insert_check(int warpID, int shared_blockID, uint64_t buffer){



// 	__shared__ vqf_block extern_blocks[WARPS_PER_BLOCK];

// 	#if DEBUG_ASSERTS

// 	assert(buffer < num_blocks);

// 	#endif


// 	uint64_t block_index = buffer;

// 	lock_block(warpID, block_index);


// 	if (warpID == 0)

// 	extern_blocks[shared_blockID] = blocks[block_index];


// 	if (!compare_blocks(blocks[block_index],extern_blocks[shared_blockID])){

// 		assert(compare_blocks(blocks[block_index],extern_blocks[shared_blockID]));

// 	}


// 	int fill_main = extern_blocks[shared_blockID].get_fill();

// 	#ifdef DEBUG_ASSERTS
// 	assert(fill_main == 0);
// 	#endif

// 	int count = FILL_CUTOFF - fill_main;

// 	int buf_size = buffer_sizes[buffer];

// 	if (buf_size < count) count = buf_size;

// 	for (int i =0; i < count; i++){

// 		#if DEBUG_ASSERTS

// 		int old_fill = extern_blocks[shared_blockID].get_fill();


// 		//relevant equation

// 		// (x mod yz) | z == x mod y?
// 		//python says no ur a dumbass this is the bug
		
// 		if (!(get_bucket_from_hash(buffers[buffer][i])  == buffer)){

// 			if (warpID == 0){

// 				printf("i %d count %d item %llu buffer %llu new_buf %llu\n", i, count, buffers[buffer][i], buffer, get_bucket_from_hash(buffers[buffer][i]));
// 			}

// 			__syncwarp();

// 			assert((buffers[buffer][i] >> TAG_BITS) % num_blocks  == buffer);

// 		}
		


// 		#endif

// 		uint64_t tag = buffers[buffer][i] & ((1ULL << TAG_BITS) -1);

// 		blocks[block_index].insert(warpID, tag);
// 		extern_blocks[shared_blockID].insert(warpID, tag);

// 		#if DEBUG_ASSERTS


// 		if (!compare_blocks(blocks[block_index],extern_blocks[shared_blockID])){

// 			assert(compare_blocks(blocks[block_index],extern_blocks[shared_blockID]));

// 		}
		

// 		assert(extern_blocks[shared_blockID].get_fill() == old_fill+1);

// 		assert(blocks[block_index].get_fill() == old_fill + 1);

// 		#endif

// 	}

// 	//write back

// 	if (!compare_blocks(blocks[block_index],extern_blocks[shared_blockID])){

// 		assert(compare_blocks(blocks[block_index],extern_blocks[shared_blockID]));

// 	}

// 	__threadfence();
// 	__syncwarp();

// 	blocks[block_index].unlock(warpID);

// 	//and decrement the count

// 	if (warpID == 0){

// 		buffers[buffer] += count;

// 		buffer_sizes[buffer] -= count;


// 	}


// }


//come back and put me in the final implementation
// __device__ bool vqf::buffer_end_dump(int warpID, uint64_t buffer){


// 	int count = buffer_sizes[buffer];

// 	for (int i =0; i < )
// }


__device__ bool optimized_vqf::query(int warpID, uint64_t key){

	uint64_t hash = hash_key(key);

	//uint64_t block_index = ((hash >> TAG_BITS) % (VIRTUAL_BUCKETS*num_blocks))/VIRTUAL_BUCKETS;
	uint64_t block_index = get_bucket_from_hash(hash);

   //this will generate a mask and get the tag bits
   //uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   //uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;
	//uint64_t alt_block_index = get_alt_hash(hash, block_index);

  //  while (block_index == alt_block_index){
		// alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
  //  }


   uint64_t team_index = block_index / BLOCKS_PER_THREAD_BLOCK;

   block_index = block_index % BLOCKS_PER_THREAD_BLOCK;


   lock_block(warpID, team_index, block_index);

   #if DEBUG_ASSERTS
 	assert(blocks[team_index].internal_blocks[block_index].assert_consistency());

 	#endif
   bool found = blocks[team_index].internal_blocks[block_index].query(warpID, hash);

   #if DEBUG_ASSERTS
   assert(blocks[team_index].internal_blocks[block_index].assert_consistency());
   #endif

  	unlock_block(warpID, team_index, block_index);

   return found;

}

__device__ bool optimized_vqf::full_query(int warpID, uint64_t key){

	uint64_t hash = hash_key(key);

	//uint64_t block_index = ((hash >> TAG_BITS) % (VIRTUAL_BUCKETS*num_blocks))/VIRTUAL_BUCKETS;
	uint64_t block_index = get_bucket_from_hash(hash);

   //this will generate a mask and get the tag bits
   //uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   //uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;
	//uint64_t alt_block_index = get_alt_hash(hash, block_index);

  //  while (block_index == alt_block_index){
		// alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
  //  }



	uint64_t teamID = block_index / BLOCKS_PER_THREAD_BLOCK;

	int internalID = block_index % BLOCKS_PER_THREAD_BLOCK;

   //lock_block(warpID, teamID, internalID);

   #if DEBUG_ASSERTS
   assert(blocks[teamID].internal_blocks[internalID].assert_consistency());

 	#endif

   bool found = blocks[teamID].internal_blocks[internalID].query(warpID, hash);

   #if DEBUG_ASSERTS
   assert(blocks[teamID].internal_blocks[internalID].assert_consistency());
   #endif

   //unlock_block(warpID, teamID, internalID);



   if (found) return true;

   //check the other block

   uint64_t alt_hash = get_alt_hash(hash, block_index);

   uint64_t alt_bucket = get_bucket_from_hash(alt_hash) % BLOCKS_PER_THREAD_BLOCK;

   if (alt_bucket == internalID) alt_bucket = (alt_bucket +1 ) % BLOCKS_PER_THREAD_BLOCK;



   //lock_block(warpID, teamID, alt_bucket);


   found = blocks[teamID].internal_blocks[alt_bucket].query(warpID, hash);

   //unlock_block(warpID, teamID, alt_bucket);

   return found;

}


//BUG: insert and remove seems to not be correct
//V1: uint64_t block_index = ((hash >> TAG_BITS) % (VIRTUAL_BUCKETS*num_blocks))/VIRTUAL_BUCKETS;
//V2: uint64_t block_index = (hash >> TAG_BITS) % num_blocks;

// __device__ bool optimized_vqf::remove(int warpID, uint64_t key){

// 	uint64_t hash = hash_key(key);


// 	uint64_t block_index = get_bucket_from_hash(hash);

//    //this will generate a mask and get the tag bits
// 	//uint64_t alt_block_index = get_alt_hash(hash, block_index);

//   //  while (block_index == alt_block_index){
// 		// alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
//   //  }

//   		lock_block(warpID, block_index);


//   		#if DEBUG_ASSERTS

//   		assert(blocks[block_index].assert_consistency());


// 		int old_fill = blocks[block_index].get_fill();

// 		//assert(blocks[block_index].assert_consistency());

// 		uint64_t md_before = blocks[block_index].md[0];


// 		#endif

//    bool found = blocks[block_index].remove(warpID, hash);


//       #if DEBUG_ASSERTS
//  		int new_fill = blocks[block_index].get_fill();

//  		//assert(blocks[block_index].assert_consistency());

//  		uint64_t md_after = blocks[block_index].md[0];

//  		if (!found){

//  			assert(md_before == md_after);

 			

//  		} else {

//  			assert(new_fill >= 0);

//  			if(old_fill-1 != new_fill){


//  				assert(blocks[block_index].assert_consistency());
//  				blocks[block_index].remove(warpID, hash);

//  				assert(old_fill-1 == new_fill);
//  			}
//  		}
 		

 		

//  		#endif

//    unlock_block(warpID, block_index);

//    //copy could be deleted from this instance

// 	 return found;

// }


// __device__ bool vqf::insert(uint64_t hash){

//    uint64_t block_index = (hash >> TAG_BITS) % num_blocks;



//    //this will generate a mask and get the tag bits
//    uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
//    uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;

//    assert(block_index < num_blocks);


//    //external locks
//    //blocks[block_index].extra_lock(block_index);
   
//    while(atomicCAS(locks + block_index, 0, 1) == 1);



//    int fill_main = blocks[block_index].get_fill();


//    if (fill_main >= SLOTS_PER_BLOCK-1){

//    	while(atomicCAS(locks + block_index, 0, 1) == 0);
//    	//blocks[block_index].unlock();

//    	return false;
//    }

//    if (fill_main < .75 * SLOTS_PER_BLOCK || block_index == alt_block_index){
//    	blocks[block_index].insert(tag);

   	

//    	int new_fill = blocks[block_index].get_fill();
//    	if (new_fill != fill_main+1){
//    		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
//    		assert(blocks[block_index].get_fill() == fill_main+1);
//    	}


//    	while(atomicCAS(locks + block_index, 1, 0) == 0);
//    	//blocks[block_index].unlock();
//    	return true;
//    }


//    while(atomicCAS(locks + block_index, 1, 0) == 0);

//    lock_blocks(block_index, alt_block_index);


//    //need to grab other block

//    //blocks[alt_block_index].extra_lock(alt_block_index);
//    while(atomicCAS(locks + alt_block_index, 0, 1) == 1);

//    int fill_alt = blocks[alt_block_index].get_fill();

//    //any larger and we can't protect metadata
//    if (fill_alt >=  SLOTS_PER_BLOCK-1){
// //   	blocks[block_index.unlock()]

//    	unlock_blocks(block_index, alt_block_index);
//    	//blocks[alt_block_index].unlock();
//    	//blocks[block_index].unlock();
//    	return false;
//    }


//    //unlock main
//    if (fill_main > fill_alt ){

//    	while(atomicCAS(locks + block_index, 1, 0) == 0);
//    	//blocks[block_index].unlock();

//    	blocks[alt_block_index].insert(tag);
//    	assert(blocks[alt_block_index].get_fill() == fill_alt+1);

//    	int new_fill = blocks[alt_block_index].get_fill();
//    	if (new_fill != fill_alt+1){
//    		printf("Broken Fill: Block %llu, old %d new %d\n", alt_block_index, fill_alt, new_fill);
//    		assert(blocks[alt_block_index].get_fill() == fill_alt+1);
//    	}

//    	while(atomicCAS(locks + alt_block_index, 1, 0) == 0);
//    	//blocks[alt_block_index].unlock();


//    } else {

//    	while(atomicCAS(locks + alt_block_index, 1, 0) == 0);
//    	//blocks[alt_block_index].unlock();
//    	blocks[block_index].insert(tag);

//    	int new_fill = blocks[block_index].get_fill();
//    	if (new_fill != fill_main+1){
//    		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
//    		assert(blocks[block_index].get_fill() == fill_main+1);
//    	}

//    	while(atomicCAS(locks + block_index, 1, 0) == 0);
//    	//blocks[block_index].unlock();

//    }


  
//    return true;



//}

__device__ uint64_t optimized_vqf::hash_key(uint64_t key){


	key = MurmurHash64A(((void *)&key), sizeof(key), seed) % ((num_blocks * VIRTUAL_BUCKETS) << TAG_BITS);

	return key;


}

__global__ void hash_all(optimized_vqf* my_vqf, uint64_t* vals, uint64_t* hashes, uint64_t nvals) {
	
	uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nvals){
		return;
	}

  uint64_t key = vals[idx];

  key = my_vqf->hash_key(key);


	//uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));


  #if DEBUG_ASSERTS

  assert(key >> TAG_BITS < my_vqf->num_blocks* VIRTUAL_BUCKETS);

  #endif
  
  hashes[idx] = key;

	return;

}



//an *optimized* version of hash all - this will force keys to be a 
__global__ void hash_all_key_purge(optimized_vqf * my_vqf, uint64_t * vals, uint64_t * hashes, uint64_t nvals){


	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = vals[tid];

	key = my_vqf->hash_key(key);

	#if TAG_BITS == 8

	uint8_t tag = key & 0xff;

	#else 

	uint16_t tag = key & 0xffff;

	#endif

	uint64_t new_key = ((my_vqf->get_bucket_from_hash(key) * VIRTUAL_BUCKETS) << TAG_BITS) + tag;


	#if DEBUG_ASSERTS


	assert(my_vqf->get_bucket_from_hash(new_key) == my_vqf->get_bucket_from_hash(key));

	#endif

	hashes[tid] = new_key;


}


//given a hashed list of inserts, convert the hashes into their alternate hash
// this is used to quickly convert the second half of the 
__global__ void alt_hash_all(optimized_vqf * my_vqf, uint64_t * vals, uint64_t nvals){



	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = vals[tid];

	key = my_vqf->get_alt_hash(key, my_vqf->get_bucket_from_hash(key));

	vals[tid] = key;
}


//set the references with tid[i] = i;
__global__ void init_references(uint64_t * vals, uint64_t nvals){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= 2*nvals) return;

	vals[tid] = tid;

}





//Bug thoughts

//keys run over range 0 - num_blocks*virtual_buckets >> tags

//to pick a slot, down shift tags to get 0-num_blocks*virtual_buckets

__global__ void set_buffers_binary(optimized_vqf * my_vqf, uint64_t num_keys, const __restrict__ uint64_t * keys){

		uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= my_vqf->num_blocks) return;

		//uint64_t slots_per_lock = VIRTUAL_BUCKETS;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		//this is fine but need to apply a hash
		uint64_t boundary = idx; //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;


		uint64_t lower = 0;
		uint64_t upper = num_keys;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			//((keys[index] >> TAG_BITS)
			uint64_t bucket = my_vqf->get_bucket_from_hash(keys[index]);


			if (index != 0)
			uint64_t old_bucket = my_vqf->get_bucket_from_hash(keys[index-1]);

			if (bucket < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

				//(get_bucket_from_hash(keys[index-1])
				//(keys[index-1] >> TAG_BITS)

			} else if (my_vqf->get_bucket_from_hash(keys[index-1]) < boundary) {

				//set index! this is the first instance where I am valid and the next isnt
				//buffers[idx] = keys+index;
				break;

			} else {

				//we are too far right, all keys to the right do not matter
				upper = index;


			}

		}

		//we either exited or have an edge condition:
		//upper == lower iff 0 or max key
		index = lower + (upper - lower)/2;

		//assert(my_vqf->get_bucket_from_hash(keys[index]) <= idx);


		my_vqf->buffers[idx] = ((uint64_t *)keys) + index;
		


}

//this can maybe be rolled into set_buffers_binary
//it performs an identical set of operations that are O(1) here
// O(log n) there, but maybe amortized

__global__ void set_buffer_lens(optimized_vqf* my_vqf, uint64_t num_keys, const __restrict__ uint64_t * keys){


	uint64_t num_buffers = my_vqf->num_blocks;


	uint64_t idx = threadIdx.x + blockDim.x*blockIdx.x;

	if (idx >= num_buffers) return;


	//only 1 thread will diverge - should be fine - any cost already exists because of tail
	if (idx != num_buffers-1){

		//this should work? not 100% convinced but it seems ok
		my_vqf->buffer_sizes[idx] = my_vqf->buffers[idx+1] - my_vqf->buffers[idx];
	} else {

		my_vqf->buffer_sizes[idx] = num_keys - (my_vqf->buffers[idx] - keys);

	}

	return;


}

__host__ uint64_t optimized_vqf::get_num_buffers(){

	uint64_t internal_num_blocks;

	cudaMemcpy(&internal_num_blocks, (uint64_t * ) this, sizeof(uint64_t), cudaMemcpyDeviceToHost);

 	cudaDeviceSynchronize();

 	return internal_num_blocks;
}

__host__ uint64_t optimized_vqf::get_num_teams(){

	uint64_t internal_num_teams;

	cudaMemcpy(&internal_num_teams, ((uint64_t * ) this) + 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);

 	cudaDeviceSynchronize();

 	return internal_num_teams;
}


//have the VQF sort the input dataset and attach the buffers to the data

__host__ void optimized_vqf::attach_buffers(uint64_t * vals, uint64_t nvals){



	hash_all_key_purge<<<(nvals - 1)/1024 + 1, 1024>>>(this, vals, vals, nvals);


	thrust::sort(thrust::device, vals, vals+nvals);




	uint64_t internal_num_blocks = get_num_buffers();
	


 	set_buffers_binary<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, nvals, vals);

 	set_buffer_lens<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, nvals, vals);


}


__global__ void vqf_block_setup(optimized_vqf * vqf){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= vqf->num_blocks) return;


 
	vqf->blocks[tid / BLOCKS_PER_THREAD_BLOCK].internal_blocks[tid % BLOCKS_PER_THREAD_BLOCK].setup();


	#if EXCLUSIVE_ACCESS

	//vqf->blocks[tid / BLOCKS_PER_THREAD_BLOCK].internal_blocks[tid % BLOCKS_PER_THREAD_BLOCK].lock(0);

	#endif

}

__host__ optimized_vqf * build_vqf(uint64_t nitems){


	#if DEBUG_ASSERTS

	printf("Debug correctness checks on. These will affect performance.\n");

	#endif

	//this seems weird but whatever
	uint64_t num_blocks = (nitems -1)/SLOTS_PER_BLOCK + 1;

	uint64_t num_teams = (num_blocks-1) / BLOCKS_PER_THREAD_BLOCK + 1;

	//rewrite num_blocks to account for any expansion.
	num_blocks = num_teams*BLOCKS_PER_THREAD_BLOCK;

	printf("Bytes used: %llu for %llu blocks.\n", num_teams*sizeof(thread_team_block),  num_blocks);

	printf("Internal block info: %llu bytes per block, %llu bytes per cache line, %llu mod, %llu slots per block, %llu blocks per thread block\n", sizeof(atomic_block), BYTES_PER_CACHE_LINE, sizeof(atomic_block) % BYTES_PER_CACHE_LINE, SLOTS_PER_BLOCK, BLOCKS_PER_THREAD_BLOCK);


	optimized_vqf * host_vqf;

	optimized_vqf * dev_vqf;

	thread_team_block * blocks;

	cudaMallocHost((void ** )& host_vqf, sizeof(optimized_vqf));

	cudaMalloc((void ** )& dev_vqf, sizeof(optimized_vqf));	

	//init host
	host_vqf->num_blocks = num_blocks;

	host_vqf->num_teams = num_teams;

	//allocate blocks
	cudaMalloc((void **)&blocks, num_teams*sizeof(thread_team_block));

	cudaMemset(blocks, 0, num_teams*sizeof(thread_team_block));

	host_vqf->blocks = blocks;


	//external locks

	//TODO: get rid of these they're not necessary
	// int * locks;

	// //numblocks or 1
	// cudaMalloc((void ** )&locks,1*sizeof(int));
	// cudaMemset(locks, 0, 1*sizeof(int));


	// host_vqf->locks = locks;


	uint64_t ** buffers;
	uint64_t * buffer_sizes;

	//in this scheme blocks are per 

	cudaMalloc((void **)& buffers, num_blocks*sizeof(uint64_t *));
	cudaMemset(buffers, 0, num_blocks*sizeof(uint64_t * ));

	cudaMalloc((void **)& buffer_sizes, num_blocks*sizeof(uint64_t));
	cudaMemset(buffer_sizes, 0, num_blocks*sizeof(uint64_t));


	host_vqf->buffers = buffers;

	host_vqf->buffer_sizes = buffer_sizes;

	host_vqf->seed = 5;


	cudaMemcpy(dev_vqf, host_vqf, sizeof(optimized_vqf), cudaMemcpyHostToDevice);

	cudaFreeHost(host_vqf);

	vqf_block_setup<<<(num_blocks - 1)/64 + 1, 64>>>(dev_vqf);
	cudaDeviceSynchronize();

	return dev_vqf;


}


//the upper (sizeof(hash - TAG_BITS)) represent the slot of the hash,
// downshift and modulus to get the slot, and then divide to get the bucket the slot belongs to
//This is outdated but still works, a maintains a guarantee that items are assigned in sorted order
//which is a precondition for the bucket inserts and power of two inserts.
__device__ uint64_t optimized_vqf::get_bucket_from_hash(uint64_t hash){

	return ((hash >> TAG_BITS) % (num_blocks * VIRTUAL_BUCKETS)) / VIRTUAL_BUCKETS;
}



//TODO: modify this to only refer to the local buckets


//generate the alternate hash for inserts, the new version requires a call to 
// get_bucket+from_hash as well, but has the additional benefit of returning a working key.
__device__ uint64_t optimized_vqf::get_alt_hash(uint64_t hash, uint64_t bucket){

	uint64_t alt_block_index = hash_key(hash);


	// while (alt_block_index == bucket){
	// 	alt_block_index = get_bucket_from_hash(hash_key(hash ^ alt_block_index));
	// }

	//ask prashant for a better way to do this it feels ridiculous.
	while (get_bucket_from_hash(alt_block_index) == bucket){

	//I goofed here and some items stall if you add just 1
	//you should jump one bucket instead, SLOTS_PER_BLOCK << TAG_BITS
	//TODO: double check this i think this maybe should be VIRTUAL_BUCKETS instead
	alt_block_index = (alt_block_index + SLOTS_PER_BLOCK << TAG_BITS);

	}

	return alt_block_index;
}





__host__ optimized_vqf * prep_host_vqf(uint64_t nitems){


	#if DEBUG_ASSERTS

	printf("Debug correctness checks on. These will affect performance.\n");

	#endif

	//this seems weird but whatever
	uint64_t num_blocks = (nitems -1)/SLOTS_PER_BLOCK + 1;

	uint64_t num_teams = (num_blocks-1) / BLOCKS_PER_THREAD_BLOCK + 1;

	//rewrite num_blocks to account for any expansion.
	num_blocks = num_teams*BLOCKS_PER_THREAD_BLOCK;

	printf("Bytes used: %llu for %llu blocks.\n", num_teams*sizeof(thread_team_block),  num_blocks);

	printf("Internal block info: %llu bytes per block, %llu bytes per cache line, %llu mod, %llu slots per block, %llu blocks per thread block\n", sizeof(atomic_block), BYTES_PER_CACHE_LINE, sizeof(atomic_block) % BYTES_PER_CACHE_LINE, SLOTS_PER_BLOCK, BLOCKS_PER_THREAD_BLOCK);


	optimized_vqf * host_vqf;

	optimized_vqf * dev_vqf;

	thread_team_block * blocks;

	cudaMallocHost((void ** )& host_vqf, sizeof(optimized_vqf));

	//cudaMalloc((void ** )& dev_vqf, sizeof(optimized_vqf));	

	//init host
	host_vqf->num_blocks = num_blocks;

	host_vqf->num_teams = num_teams;

	//allocate blocks
	cudaMallocHost((void **)&blocks, num_teams*sizeof(thread_team_block));

	//cudaMemset(blocks, 0, num_teams*sizeof(thread_team_block));

	host_vqf->blocks = blocks;


	//external locks

	//TODO: get rid of these they're not necessary
	// int * locks;

	// //numblocks or 1
	// cudaMalloc((void ** )&locks,1*sizeof(int));
	// cudaMemset(locks, 0, 1*sizeof(int));


	// host_vqf->locks = locks;


	uint64_t ** buffers;
	uint64_t * buffer_sizes;

	//in this scheme blocks are per 

	cudaMallocHost((void **)& buffers, num_blocks*sizeof(uint64_t *));
	//cudaMemset(buffers, 0, num_blocks*sizeof(uint64_t * ));

	cudaMallocHost((void **)& buffer_sizes, num_blocks*sizeof(uint64_t));
	//cudaMemset(buffer_sizes, 0, num_blocks*sizeof(uint64_t));


	host_vqf->buffers = buffers;

	host_vqf->buffer_sizes = buffer_sizes;

	host_vqf->seed = 5;


	//cudaMemcpy(dev_vqf, host_vqf, sizeof(optimized_vqf), cudaMemcpyHostToDevice);

	//cudaFreeHost(host_vqf);

	//vqf_block_setup<<<(num_blocks - 1)/64 + 1, 64>>>(dev_vqf);
	cudaDeviceSynchronize();

	return host_vqf;


}


__host__ void optimized_vqf::insert_async(uint64_t * items, uint64_t nitems, uint64_t num_teams, uint64_t num_blocks, cudaStream_t stream, uint64_t * misses){


	//modified attach buffers
	//assume sorted so we just need to attach the buffers
	set_buffers_binary<<< (num_blocks -1) / 1024 + 1, 1024, 0, stream>>>(this, nitems, items);
	set_buffer_lens<<< (num_blocks -1) / 1024 + 1, 1024, 0, stream>>>(this, nitems, items);

	//should this be <<<num_teams -1 / BLOCK_SIZE + 1>>>?
	bulk_insert_kernel<<<num_teams, BLOCK_SIZE, 0, stream>>>(this, misses);





}


__global__ void average_count(optimized_vqf * vqf, uint64_t * counter, uint64_t num_blocks){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid >= num_blocks) return;

	uint64_t blockID = tid / BLOCKS_PER_THREAD_BLOCK;

	int sub_blockID = tid % BLOCKS_PER_THREAD_BLOCK;

	uint64_t fill = vqf->blocks[blockID].internal_blocks[sub_blockID].get_fill();

	atomicAdd((unsigned long long int *) counter, (unsigned long long int) fill);


}


__global__ void average_count_teams(optimized_vqf * vqf, uint64_t * counter, uint64_t * max, uint64_t * min, uint64_t num_teams){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid >= num_teams) return;

	uint64_t internal_counter = 0;
	
	for (int i=0; i < BLOCKS_PER_THREAD_BLOCK; i++){

		internal_counter += vqf->blocks[tid].internal_blocks[i].get_fill();


	}



	
	atomicAdd((unsigned long long int *) counter, (unsigned long long int) internal_counter);

	atomicMax((unsigned long long int *) max, (unsigned long long int) internal_counter);

	atomicMin((unsigned long long int *) min, (unsigned long long int) internal_counter);


}


//find and print the average fill per atomic_block
__host__ void optimized_vqf::get_average_fill_block(){

	uint64_t * count;

	cudaMallocManaged((void **)& count, sizeof(uint64_t));


	uint64_t internal_num_blocks = get_num_buffers();

	count[0] = 0;

	cudaDeviceSynchronize();


	average_count<<<(internal_num_blocks-1)/1024 +1, 1024>>>(this, count, internal_num_blocks);

	cudaDeviceSynchronize();


	uint64_t average = count[0];

	double float_average = 1.0*average / internal_num_blocks;

	printf("Average occupancy per array: %f/%llu: %f\n", float_average, SLOTS_PER_BLOCK, float_average/SLOTS_PER_BLOCK);

	cudaFree(count);

}


//find and print the average fill per atomic_block
__host__ void optimized_vqf::get_average_fill_team(){

	uint64_t * count;

	cudaMallocManaged((void **)& count, sizeof(uint64_t));

	uint64_t * max_count;

	cudaMallocManaged((void **)& max_count, sizeof(uint64_t));

	uint64_t * min_count;

	cudaMallocManaged((void **)& min_count, sizeof(uint64_t));

	




	uint64_t internal_num_teams = get_num_teams();

	count[0] = 0;

	max_count[0] = 0;

	min_count[0] = SLOTS_PER_BLOCK*BLOCKS_PER_THREAD_BLOCK;

	cudaDeviceSynchronize();


	average_count_teams<<<(internal_num_teams-1)/1024 +1, 1024>>>(this, count, max_count, min_count, internal_num_teams);

	cudaDeviceSynchronize();


	uint64_t average = count[0];

	double float_average = 1.0*average / internal_num_teams;

	printf("Average occupancy per thread_team_block: %f/%llu: %f\n", float_average, BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK, float_average/(BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK));

	printf("Max occupancy per thread_team_block: %llu/%llu: %f\n", max_count[0], BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK, 1.0*max_count[0]/(BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK));

	printf("Min occupancy per thread_team_block: %llu/%llu: %f\n", min_count[0], BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK, 1.0*min_count[0]/(BLOCKS_PER_THREAD_BLOCK*SLOTS_PER_BLOCK));



	cudaFree(count);

	cudaFree(max_count);

	cudaFree(min_count);

}



//each thread team sorts one block
__global__ void vqf_sort_kernel(optimized_vqf * vqf){


	//CHANGE HERE

	// uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	// uint64_t teamID = tid / (BLOCK_SIZE);


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	int warpID = tid % 32;

	//clip to inside block
	uint64_t blockID = ((tid) / 32) % WARPS_PER_BLOCK;



	uint64_t teamID = tid / (BLOCK_SIZE);

	if (teamID >= vqf->num_teams) return;


	//short_warp_sort(vqf->blocks[teamID].internal_blocks[blockID].tags, vqf->blocks[teamID].internal_blocks[blockID].get_fill(),blockID, warpID);


	vqf->blocks[teamID].internal_blocks[blockID].sort_block(blockID, warpID);

	__threadfence();

	__syncwarp();



	if (warpID == 0){

		if (!vqf->blocks[teamID].internal_blocks[blockID].assert_sorted(warpID)){


			assert(vqf->blocks[teamID].internal_blocks[blockID].assert_sorted(warpID));
	

	}

	}






}

__host__ void optimized_vqf::sort_and_check(){

	uint64_t num_buffers = get_num_buffers();

	vqf_sort_kernel<<<(num_buffers -1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(this);


}

#endif

