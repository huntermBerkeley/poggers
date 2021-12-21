
#ifndef OPTIMIZED_VQF_C
#define OPTIMIZED_VQF_C


#include <cuda.h>
#include <cuda_runtime_api.h>


#include "include/optimized_vqf.cuh"
#include "include/gpu_block.cuh"
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



struct is_tombstone
{
	__host__ __device__ bool operator()(const uint64_t val){

		return val == TOMBSTONE;
	}
};


__device__ void optimized_vqf::lock_block(int warpID, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 0,1) != 0);	
	// }
	// __syncwarp();

	//TODO: turn me back on

	#if EXCLUSIVE_ACCESS


	#else 
	blocks[lock].lock(warpID);

	#endif
}

__device__ void optimized_vqf::unlock_block(int warpID, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 1,0) != 1);	

	// }

	// __syncwarp();

	#if EXCLUSIVE_ACCESS


	#else

	blocks[lock].unlock(warpID);

	#endif


}

__device__ void optimized_vqf::lock_blocks(int warpID, uint64_t lock1, uint64_t lock2){


	if (lock1 < lock2){

		lock_block(warpID, lock1);
		lock_block(warpID, lock2);
		//while(atomicCAS(locks + lock2, 0,1) == 1);

	} else {


		lock_block(warpID, lock2);
		lock_block(warpID, lock1);
		
	}

	


}

__device__ void optimized_vqf::unlock_blocks(int warpID, uint64_t lock1, uint64_t lock2){


	if (lock1 > lock2){

		unlock_block(warpID, lock1);
		unlock_block(warpID, lock2);
		
	} else {

		unlock_block(warpID, lock2);
		unlock_block(warpID, lock1);
	}
	

}

__device__ bool optimized_vqf::insert(int warpID, uint64_t key, bool hashed){


	uint64_t hash;

	if (hashed){

		hash = key;

	} else {

		hash = hash_key(key);


	}

   uint64_t block_index = get_bucket_from_hash(hash);

   //uint64_t alt_block_index = get_alt_hash(hash, block_index);



 	lock_block(warpID, block_index);

   int fill_main = blocks[block_index].get_fill();



   bool toReturn = false;


   	if (fill_main < MAX_FILL){
   		blocks[block_index].insert(warpID, hash);

   		toReturn = true;


   		#if DEBUG_ASSERTS
   		int new_fill = blocks[block_index].get_fill();
   		if (new_fill != fill_main+1){

   		//blocks[block_index].printMetadata();
   		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
   		assert(blocks[block_index].get_fill() == fill_main+1);
   		}

   		assert(blocks[block_index].query(warpID, hash));
   		#endif

   	}

   unlock_block(warpID, block_index);



   return toReturn;





}


//acts like a bulked insert, will dump into the main buffer up to some preset fill ratio
//likely to be the same value as the optimized one, 20.

__device__ bool optimized_vqf::buffer_insert(int warpID, uint64_t buffer){


	#if DEBUG_ASSERTS

	assert(buffer < num_blocks);

	#endif


	uint64_t block_index = buffer;

	lock_block(warpID, block_index);

	int fill_main = blocks[block_index].get_fill();

	int count = FILL_CUTOFF - fill_main;

	int buf_size = buffer_sizes[buffer];

	if (buf_size < count) count = buf_size;


	blocks[block_index].bulk_insert(warpID, buffers[buffer], count);
	
	#if EXCLUSIVE_ACCESS

	#else
	blocks[block_index].unlock(warpID);

	#endif

	//and decrement the count

	if (warpID == 0){

		buffers[buffer] += count;

		buffer_sizes[buffer] -= count;


	}


	return true;


}

__device__ void optimized_vqf::multi_buffer_insert(int warpID, int init_blockID, uint64_t start_buffer){


	__shared__ gpu_block extern_blocks[WARPS_PER_BLOCK*REGIONS_PER_WARP];


	#if DEBUG_ASSERTS

	assert(start_buffer < num_blocks);

	#endif


	int shared_blockID = init_blockID * REGIONS_PER_WARP;



	if (start_buffer + warpID < num_blocks)

	{


		extern_blocks[shared_blockID + warpID % REGIONS_PER_WARP] = blocks[start_buffer + warpID % REGIONS_PER_WARP];

	}

	__syncwarp();

	#if EXCLUSIVE_ACCESS


	#else

	for (int i = 0; i < REGIONS_PER_WARP; i++){

		if (start_buffer + i >= num_blocks) break;


		extern_blocks[shared_blockID + i].lock(warpID);

	
	}


	// 	


	// }

	__syncwarp();

	#endif
	

	for (int i = 0; i < REGIONS_PER_WARP; i++){

		if (start_buffer + i >= num_blocks) break;

		int extern_id = shared_blockID + i;

		uint64_t buffer = start_buffer + i;



		int fill_main = extern_blocks[extern_id].get_fill();

		#ifdef DEBUG_ASSERTS
		assert(fill_main == 0);
		#endif

		int count = FILL_CUTOFF - fill_main;

		int buf_size = buffer_sizes[buffer];

		if (buf_size < count) count = buf_size;

		extern_blocks[extern_id].bulk_insert(warpID, buffers[buffer], count); 


	//wrap up the loops

	#if EXCLUSIVE_ACCESS

		

	#else

		extern_blocks[extern_id].unlock(warpID);

	#endif

	if (warpID == 0){

		buffers[buffer] += count;

		buffer_sizes[buffer] -= count;


	}

	__syncwarp();

	}


	//write back

	// for (int i = 0; i < REGIONS_PER_WARP; i++){

	// 	if (start_buffer + i >= num_blocks) break;

		
	// 	extern_blocks[shared_blockID + i].unlock(warpID);
	// }


	if (start_buffer + warpID < num_blocks) {


		blocks[start_buffer + (warpID % REGIONS_PER_WARP)] = extern_blocks[ shared_blockID + (warpID % REGIONS_PER_WARP)];


	}
	//if (warpID == 0)
	//blocks[block_index] = extern_blocks[shared_blockID];

	//blocks[block_index].load_block(warpID, extern_blocks + shared_blockID);


	__threadfence();
	__syncwarp();

	//blocks[block_index].unlock(warpID);
	

	//and decrement the count


}


__device__ int optimized_vqf::buffer_query(int warpID, uint64_t buffer){


	#if DEBUG_ASSERTS

	assert(buffer < num_blocks);

	#endif


	uint64_t block_index = buffer;

	lock_block(warpID, block_index);

	
	int buf_size = buffer_sizes[buffer];


	int found = blocks[block_index].bulk_query(warpID, buffers[buffer], buf_size);
	

	unlock_block(warpID, block_index);

	//and decrement the count



	return buf_size - found;


}




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



   lock_block(warpID, block_index);

   #if DEBUG_ASSERTS
 	assert(blocks[block_index].assert_consistency());

 	#endif
   bool found = blocks[block_index].query(warpID, hash);

   #if DEBUG_ASSERTS
   assert(blocks[block_index].assert_consistency());
   #endif

  	unlock_block(warpID, block_index);

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



   lock_block(warpID, block_index);

   #if DEBUG_ASSERTS
 	assert(blocks[block_index].assert_consistency());

 	#endif
   bool found = blocks[block_index].query(warpID, hash);

   #if DEBUG_ASSERTS
   assert(blocks[block_index].assert_consistency());
   #endif

  	unlock_block(warpID, block_index);



  	if (found) return true;

   //check the other block

  	uint64_t alt_hash = get_alt_hash(hash, block_index);

   uint64_t alt_block_index = get_bucket_from_hash(alt_hash);

   lock_block(warpID, alt_block_index);


   found = blocks[alt_block_index].query(warpID, alt_hash);

   unlock_block(warpID, alt_block_index);

   return found;
}


//BUG: insert and remove seems to not be correct
//V1: uint64_t block_index = ((hash >> TAG_BITS) % (VIRTUAL_BUCKETS*num_blocks))/VIRTUAL_BUCKETS;
//V2: uint64_t block_index = (hash >> TAG_BITS) % num_blocks;

__device__ bool optimized_vqf::remove(int warpID, uint64_t key){

	uint64_t hash = hash_key(key);


	uint64_t block_index = get_bucket_from_hash(hash);

   //this will generate a mask and get the tag bits
	//uint64_t alt_block_index = get_alt_hash(hash, block_index);

  //  while (block_index == alt_block_index){
		// alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
  //  }

  		lock_block(warpID, block_index);


  		#if DEBUG_ASSERTS

  		assert(blocks[block_index].assert_consistency());


		int old_fill = blocks[block_index].get_fill();

		//assert(blocks[block_index].assert_consistency());

		uint64_t md_before = blocks[block_index].md[0];


		#endif

   bool found = blocks[block_index].remove(warpID, hash);


      #if DEBUG_ASSERTS
 		int new_fill = blocks[block_index].get_fill();

 		//assert(blocks[block_index].assert_consistency());

 		uint64_t md_after = blocks[block_index].md[0];

 		if (!found){

 			assert(md_before == md_after);

 			

 		} else {

 			assert(new_fill >= 0);

 			if(old_fill-1 != new_fill){


 				assert(blocks[block_index].assert_consistency());
 				blocks[block_index].remove(warpID, hash);

 				assert(old_fill-1 == new_fill);
 			}
 		}
 		

 		

 		#endif

   unlock_block(warpID, block_index);

   //copy could be deleted from this instance

	 return found;

}


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


//Bug thoughts

//keys run over range 0 - num_blocks*virtual_buckets >> tags

//to pick a slot, down shift tags to get 0-num_blocks*virtual_buckets

__global__ void set_buffers_binary(optimized_vqf * my_vqf, uint64_t num_keys, uint64_t * keys){

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


		my_vqf->buffers[idx] = keys + index;
		


}

//this can maybe be rolled into set_buffers_binary
//it performs an identical set of operations that are O(1) here
// O(log n) there, but maybe amortized

__global__ void set_buffer_lens(optimized_vqf* my_vqf, uint64_t num_keys, uint64_t * keys){


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
 
	vqf->blocks[tid].setup();


	#if EXCLUSIVE_ACCESS

	vqf->blocks[tid].lock(0);

	#endif

}

__host__ optimized_vqf * build_vqf(uint64_t nitems){


	#if DEBUG_ASSERTS

	printf("Debug correctness checks on. These will affect performance.\n");

	#endif

	//this seems weird but whatever
	uint64_t num_blocks = (nitems -1)/SLOTS_PER_BLOCK + 1;


	printf("Bytes used: %llu for %llu blocks.\n", num_blocks*sizeof(gpu_block),  num_blocks);


	optimized_vqf * host_vqf;

	optimized_vqf * dev_vqf;

	gpu_block * blocks;

	cudaMallocHost((void ** )& host_vqf, sizeof(optimized_vqf));

	cudaMalloc((void ** )& dev_vqf, sizeof(optimized_vqf));	

	//init host
	host_vqf->num_blocks = num_blocks;

	//allocate blocks
	cudaMalloc((void **)&blocks, num_blocks*sizeof(gpu_block));

	cudaMemset(blocks, 0, num_blocks*sizeof(gpu_block));

	host_vqf->blocks = blocks;


	//external locks
	int * locks;
	cudaMalloc((void ** )&locks, num_blocks*sizeof(int));
	cudaMemset(locks, 0, num_blocks*sizeof(int));


	host_vqf->locks = locks;


	uint64_t ** buffers;
	uint64_t * buffer_sizes;

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




//generate the alternate hash for inserts, the new version requires a call to 
// get_bucket+from_hash as well, but has the additional benefit of returning a working key.
__device__ uint64_t optimized_vqf::get_alt_hash(uint64_t hash, uint64_t bucket){

	uint64_t alt_block_index = hash_key(hash);


	// while (alt_block_index == bucket){
	// 	alt_block_index = get_bucket_from_hash(hash_key(hash ^ alt_block_index));
	// }

	//ask prashant for a better way to do this it feels ridiculous.
	while (get_bucket_from_hash(alt_block_index) == bucket){

	alt_block_index = (alt_block_index + 1);

	}

	return alt_block_index;
}


//POWER_OF_TWO_HASH
//The following segments of code allow the vqf to perform bulked power-of-two-choice hashing
//This is performed in iterations to allow for self-balancing - every iteration, each item preps
//to be inserted into one of two 
__global__ void generate_hashes_and_references(optimized_vqf * vqf, uint64_t nvals, uint64_t * vals, uint64_t * combined_hashes, uint64_t * combined_references){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;


	uint64_t hash = vqf->hash_key(vals[tid]);


	
	combined_hashes[tid] = hash;

	combined_hashes[tid + nvals] = vqf->get_alt_hash(hash, vqf->get_bucket_from_hash(hash));

	combined_references[tid] = tid;

	combined_references[tid + nvals] = tid + nvals;


}

//using the references, set a 
//references aren't setting correctly
__global__ void set_references_from_buckets(optimized_vqf * vqf, uint64_t nvals, uint64_t * combined_hashes, uint64_t * combined_references, uint8_t * firsts, uint8_t * seconds){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t teamID = tid / 32;

	int warpID = tid % 32;


	if (teamID >= vqf->num_blocks) return;

	uint64_t bucket_offset = vqf->buffers[teamID] - combined_hashes;


	//if (bucket_offset > vqf->num_blocks)

	int bucket_max = vqf->buffer_sizes[teamID];


	for(int i=warpID; i < bucket_max; i+= 32){


		//assign the items back based on i
		uint64_t reference = combined_references[bucket_offset + i];


		if (reference >= nvals){

			reference -= nvals;

			seconds[reference] = i;
		} else {

			firsts[reference] = i;
		}

		// if (reference >= nvals){
		// 	reference -= nvals;
		// 	loc = seconds;
		// }

		// #if DEBUG_ASSERTS

		// assert(reference < nvals);

		// #endif

		// loc[reference] = i;
	}


}

//now that the references are set up, each item should tombstone the larger of it's inserts
// how to andle the empty case? For now, successes should tombstone themselves 
__global__ void tombstone_from_references(optimized_vqf * vqf, uint64_t nvals, uint64_t * combined_hashes, uint64_t * combined_references, uint8_t * firsts, uint8_t * seconds){



	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t teamID = tid / 32;

	int warpID = tid % 32;

	if (teamID >= vqf->num_blocks) return;

	uint64_t bucket_offset = vqf->buffers[teamID] - combined_hashes;


	
	int bucket_max = vqf->buffer_sizes[teamID];


	for(int i=warpID; i < bucket_max; i+= 32){

		uint64_t reference = combined_references[bucket_offset + i];


		//longer unwrapped version of hte logic used in tombstoning
		if (reference >= nvals){

			//secondary key
			reference -= nvals;

			if (seconds[reference] >= firsts[reference]){

				combined_hashes[bucket_offset + i] = TOMBSTONE;
				

			}

		} else {

			//primary key

			if (firsts[reference] > seconds[reference]){

				combined_hashes[bucket_offset + i] = TOMBSTONE;

			}


		}

		// if (reference == TOMBSTONE) printf("weird bug!\n");

		// uint8_t * primary_loc = firsts;
		// uint8_t * secondary_loc = seconds;

		// if (reference >= nvals){

		// 	reference -= nvals;
		// 	primary_loc = seconds;
		// 	secondary_loc = firsts;


		// }

		// #if DEBUG_ASSERTS

		// assert(reference < nvals);

		// #endif

		// //rely on warp optimizer to sync

		// //if the values are equal, make sure to drop secondary


		// if (primary_loc[reference] >= secondary_loc[reference]){

		// 	//secondary location is smaller, if valid we should tombstone ourselves
		// 	if (secondary_loc[reference] < MAX_FILL){

		// 		//secondary_loc is being written!
		// 		 //combined_references[bucket_offset + i] = TOMBSTONE;
		// 		 combined_hashes[bucket_offset + i] = TOMBSTONE;

		// 	}

		// }
		



	}



}




//match references to hashes
__global__ void purge_references(uint64_t nvals, uint64_t * combined_hashes, uint64_t * combined_references){

	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= nvals) return;

	if (combined_hashes[tid] == TOMBSTONE) combined_references[tid] = TOMBSTONE;

}


// //run some assertion checks that the buffers are accurrately dropping the correct value
// __global__ void check_bucket_tombstones(optimized_vqf * vqf, uint64_t nvals, uint64_t * combined_hashes, uint64_t * combined_references, uint8_t * firsts, uint8_t * seconds){

// 	uint64_t 



// }


//we can estimate the number of items that need to be tombstoned
__global__ void count_tombstones(uint64_t nvals, uint8_t * firsts, uint8_t * seconds, uint64_t * counter){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	if (firsts[tid] < MAX_FILL || seconds[tid] < MAX_FILL){

		atomicAdd((unsigned long long int *) counter, 1ULL);
	}

}


//precondition - buffers are formatted and prepper for the insert
__global__ void bulk_buffer_insert(optimized_vqf * vqf){


	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	uint64_t teamID = tid/32;

	int warpID = tid % 32;

	if (teamID >= vqf->num_blocks) return;

	vqf->buffer_insert(warpID, teamID);


}


//sort and merge

__host__ void optimized_vqf::insert_power_of_two(uint64_t * vals, uint64_t nvals){

	uint64_t * combined_hashes;


	uint64_t * combined_references;

	uint8_t  * firsts;

	uint8_t * seconds;


	cudaMalloc((void **)&combined_hashes, sizeof(uint64_t)*2*nvals);

	cudaMalloc((void **)&combined_references, sizeof(uint64_t)*2*nvals);

	cudaMalloc((void **)&firsts, sizeof(uint8_t)*nvals);
	cudaMalloc((void **)&seconds, sizeof(uint8_t)*nvals);


	uint64_t * counter;

	cudaMallocManaged((void **)&counter, sizeof(uint64_t));


	generate_hashes_and_references<<<(nvals-1)/POWER_BLOCK_SIZE + 1, POWER_BLOCK_SIZE>>>(this, nvals, vals, combined_hashes, combined_references);

	cudaDeviceSynchronize();

	counter[0] = 0;



	thrust::sort_by_key(thrust::device, combined_hashes, combined_hashes+nvals*2, combined_references);

	cudaDeviceSynchronize();


	//need to modify assign_buffers to use buffers as index
	uint64_t internal_num_blocks = get_num_buffers();
	


 	set_buffers_binary<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, 2*nvals, combined_hashes);

 	set_buffer_lens<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, 2*nvals, combined_hashes);


 
	set_references_from_buckets<<<(internal_num_blocks*32 -1)/POWER_BLOCK_SIZE + 1, POWER_BLOCK_SIZE>>>(this, nvals, combined_hashes, combined_references, firsts, seconds);
	

	tombstone_from_references<<<(internal_num_blocks*32 -1)/POWER_BLOCK_SIZE + 1, POWER_BLOCK_SIZE>>>(this, nvals, combined_hashes, combined_references, firsts, seconds);
	

	purge_references<<<(2*nvals -1) / POWER_BLOCK_SIZE + 1, POWER_BLOCK_SIZE>>>(2*nvals, combined_hashes, combined_references);



	cudaDeviceSynchronize();
	//struct required for power_of_two_bulk
	//has been moved outside of functions so it can be referenced "globally"
	// struct declared inside of hust func cannot be dereferenced in global

	//count_tombstones<<<(nvals -1)/ POWER_BLOCK_SIZE +1, POWER_BLOCK_SIZE>>>(nvals, firsts, seconds, counter);


	//cudaDeviceSynchronize();

	//printf("Count: %llu \n", counter[0]);


	//wrap in device vector to force thurst::device?
	thrust::device_ptr<uint64_t> combined_hashes_thrust_ptr = thrust::device_pointer_cast(combined_hashes);

	thrust::device_ptr<uint64_t> combined_references_thrust_ptr = thrust::device_pointer_cast(combined_references);


	thrust::device_ptr<uint64_t> items_end_thrust = thrust::remove_if(combined_hashes_thrust_ptr, combined_hashes_thrust_ptr + 2*nvals, is_tombstone());
	thrust::device_ptr<uint64_t> refs_end_thrust = thrust::remove_if(combined_references_thrust_ptr, combined_references_thrust_ptr + 2*nvals, is_tombstone());


	uint64_t * items_end = thrust::raw_pointer_cast(items_end_thrust);
   uint64_t * refs_end = thrust::raw_pointer_cast(refs_end_thrust);


	uint64_t new_nvals = items_end - combined_hashes;
	uint64_t new_rev_vals = refs_end - combined_references;

	printf("New nvals / old nvals: %llu / %llu \n", new_nvals, nvals);

	printf("Ref vals / old nvals: %llu / %llu \n", new_rev_vals, nvals);

	//buggg - exactly one ref is gonna be weird - cause it points to 0/
	assert(new_nvals == new_rev_vals);

	if (new_nvals != nvals){

		printf("Not all items could be cleanly inserted - add multiple looks fool.\n");
		abort();


	}


	cudaFree(firsts);
	cudaFree(seconds);


	//reattach buffers
	set_buffers_binary<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, nvals, combined_hashes);

 	set_buffer_lens<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, nvals, combined_hashes);


 	bulk_buffer_insert<<<(internal_num_blocks*32 -1)/POWER_BLOCK_SIZE + 1, POWER_BLOCK_SIZE>>>(this);


 	cudaDeviceSynchronize();


 	cudaFree(combined_hashes);

 	cudaFree(combined_references);




}










#endif

