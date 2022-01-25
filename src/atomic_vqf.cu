
#ifndef ATOMIC_VQF_C
#define ATOMIC_VQF_C


#include <cuda.h>
#include <cuda_runtime_api.h>


#include "include/atomic_vqf.cuh"
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


struct is_tombstone
{
	__host__ __device__ bool operator()(const uint64_t val){

		return val == TOMBSTONE;
	}
};
 

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


	//DEBUGGING - This was too small - resulted in multiple threads dropping early
	//nt sure if this is the only bug
	uint64_t teamID = tid / (BLOCK_SIZE);



	//TODO double check me
	if (teamID >= vqf->num_teams) return;


	vqf->mini_filter_block(misses);

	return;

	


}

__global__ void bulk_query_kernel(optimized_vqf * vqf, uint64_t * items, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t teamID = tid / (BLOCK_SIZE);

	if (teamID >= vqf->num_teams) return;

	vqf->mini_filter_queries(items, hits);
}



//attach buffers, create thread groups, and launch
__host__ void optimized_vqf::bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses){



	uint64_t num_teams = get_num_teams();


	attach_buffers(items, nitems);

	bulk_insert_kernel<<<num_teams, BLOCK_SIZE>>>(this, misses);



}



//speed up queries by sorting to minimize memory movements required.
// This will sort the query list but won't mutate the items themselves
__host__ void optimized_vqf::bulk_query(uint64_t * items, uint64_t nitems, bool * hits){


	uint64_t num_teams = get_num_teams();

	attach_buffers(items, nitems);

	bulk_query_kernel<<<num_teams, BLOCK_SIZE>>>(this, items, hits);


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

}


__device__ bool optimized_vqf::query_single_buffer_block(thread_team_block * local_blocks, uint64_t blockID, int warpID, int threadID, uint64_t * items, bool * hits){


	#if DEBUG_ASSERTS

	assert(blockID < num_teams);

	assert((blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);

	#endif

	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + warpID;


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

			if (alt_bucket == warpID) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;



			bool found = local_blocks->internal_blocks[alt_bucket].query(threadID, hash);

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

   lock_block(warpID, teamID, internalID);

   #if DEBUG_ASSERTS
 	assert(blocks[teamID].internal_blocks[internalID].assert_consistency());

 	#endif
   bool found = blocks[teamID].internal_blocks[internalID].query(warpID, hash);

   #if DEBUG_ASSERTS
   assert(blocks[teamID].internal_blocks[internalID].assert_consistency());
   #endif

   unlock_block(warpID, teamID, internalID);



   if (found) return true;

   //check the other block

   uint64_t alt_hash = get_alt_hash(hash, block_index);

   uint64_t alt_bucket = get_bucket_from_hash(alt_hash) % BLOCKS_PER_THREAD_BLOCK;

   if (alt_bucket == internalID) alt_bucket = (alt_bucket +1 ) % BLOCKS_PER_THREAD_BLOCK;



   lock_block(warpID, teamID, alt_bucket);


   found = blocks[teamID].internal_blocks[alt_bucket].query(warpID, hash);

   unlock_block(warpID, teamID, alt_bucket);

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

		assert(my_vqf->get_bucket_from_hash(keys[index]) == idx);


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



	hash_all<<<(nvals - 1)/1024 + 1, 1024>>>(this, vals, vals, nvals);


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

#endif

