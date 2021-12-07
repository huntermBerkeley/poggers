

#ifndef VQF_C
#define VQF_C



#include "include/single_vqf.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/vqf_team_block.cuh"

#include <iostream>

#include <fstream>
#include <assert.h>

//Thrust Sorting
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "include/hashutil.cuh"


#define DEBUG_ASSERTS 1
#define MAX_FILL 28
#define SINGLE_REGION 0
#define FILL_CUTOFF 24

__device__ void vqf::lock_block(int warpID, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 0,1) != 0);	
	// }
	// __syncwarp();

	//TODO: turn me back on
	blocks[lock].lock(warpID);
}

__device__ void vqf::unlock_block(int warpID, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 1,0) != 1);	

	// }

	// __syncwarp();

	blocks[lock].unlock(warpID);
}

__device__ void vqf::lock_blocks(int warpID, uint64_t lock1, uint64_t lock2){


	if (lock1 < lock2){

		lock_block(warpID, lock1);
		lock_block(warpID, lock2);
		//while(atomicCAS(locks + lock2, 0,1) == 1);

	} else {


		lock_block(warpID, lock2);
		lock_block(warpID, lock1);
		
	}

	


}

__device__ void vqf::unlock_blocks(int warpID, uint64_t lock1, uint64_t lock2){


	if (lock1 > lock2){

		unlock_block(warpID, lock1);
		unlock_block(warpID, lock2);
		
	} else {

		unlock_block(warpID, lock2);
		unlock_block(warpID, lock1);
	}
	

}

__device__ bool vqf::insert(int warpID, uint64_t key){


	uint64_t hash = hash_key(key);

   uint64_t block_index = get_bucket_from_hash(hash);



   //this will generate a mask and get the tag bits
   uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;

   // assert(block_index < num_blocks);


   //external locks
   //blocks[block_index].extra_lock(block_index);


   // while (block_index == alt_block_index){
   // 	alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
   // }
 	



 	lock_block(warpID, block_index);

   int fill_main = blocks[block_index].get_fill();



   bool toReturn = false;


   	if (fill_main < MAX_FILL){
   		blocks[block_index].insert(warpID, tag);

   		toReturn = true;


   		#if DEBUG_ASSERTS
   		int new_fill = blocks[block_index].get_fill();
   		if (new_fill != fill_main+1){

   		blocks[block_index].printMetadata();
   		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
   		assert(blocks[block_index].get_fill() == fill_main+1);
   		}

   		assert(blocks[block_index].query(warpID, tag));
   		#endif

   	}

   	unlock_block(warpID, block_index);



   return toReturn;





}


//acts like a bulked insert, will dump into the main buffer up to some preset fill ratio
//likely to be the same value as the optimized one, 20.

__device__ bool vqf::buffer_insert(int warpID, uint64_t buffer){


	#if DEBUG_ASSERTS

	assert(buffer < num_blocks);

	#endif


	uint64_t block_index = buffer;

	lock_block(warpID, block_index);

	int fill_main = blocks[block_index].get_fill();

	int count = FILL_CUTOFF - fill_main;

	int buf_size = buffer_sizes[buffer];

	if (buf_size < count) count = buf_size;

	for (int i =0; i < count; i++){

		#if DEBUG_ASSERTS

		int old_fill = blocks[block_index].get_fill();


		//relevant equation

		// (x mod yz) | z == x mod y?
		//python says no ur a dumbass this is the bug
		
		if (!(get_bucket_from_hash(buffers[buffer][i])  == buffer)){

			if (warpID == 0){

				printf("i %d count %d item %llu buffer %llu new_buf %llu\n", i, count, buffers[buffer][i], buffer, get_bucket_from_hash(buffers[buffer][i]));
			}

			__syncwarp();

			assert((buffers[buffer][i] >> TAG_BITS) % num_blocks  == buffer);

		}
		


		#endif

		uint64_t tag = buffers[buffer][i] & ((1ULL << TAG_BITS) -1);
		blocks[block_index].insert(warpID, tag);

		#if DEBUG_ASSERTS

		assert(blocks[block_index].get_fill() == old_fill+1);

		#endif

	}

	blocks[block_index].unlock(warpID);

	//and decrement the count

	if (warpID == 0){

		buffers[buffer] += count;

		buffer_sizes[buffer] -= count;


	}


}


__device__ bool vqf::query(int warpID, uint64_t key){

	uint64_t hash = hash_key(key);

	//uint64_t block_index = ((hash >> TAG_BITS) % (VIRTUAL_BUCKETS*num_blocks))/VIRTUAL_BUCKETS;
	uint64_t block_index = get_bucket_from_hash(hash);

   //this will generate a mask and get the tag bits
   uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;


  //  while (block_index == alt_block_index){
		// alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
  //  }



   	lock_block(warpID, block_index);


 
   	bool found = blocks[block_index].query(warpID, tag);


	   unlock_block(warpID, block_index);

   return found;

}

//BUG: insert and remove seems to not be correct
//V1: uint64_t block_index = ((hash >> TAG_BITS) % (VIRTUAL_BUCKETS*num_blocks))/VIRTUAL_BUCKETS;
//V2: uint64_t block_index = (hash >> TAG_BITS) % num_blocks;

__device__ bool vqf::remove(int warpID, uint64_t key){

	uint64_t hash = hash_key(key);


	uint64_t block_index = get_bucket_from_hash(hash);

   //this will generate a mask and get the tag bits
   uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;

  //  while (block_index == alt_block_index){
		// alt_block_index = (alt_block_index * (tag * 0x5bd1e995)) % num_blocks;
  //  }

   lock_block(warpID, block_index);


  		#if DEBUG_ASSERTS
		int old_fill = blocks[block_index].get_fill();

		assert(blocks[block_index].assert_consistency());

		uint64_t md_before = blocks[block_index].md[0];


		#endif

   bool found = blocks[block_index].remove(warpID, tag);


      #if DEBUG_ASSERTS
 		int new_fill = blocks[block_index].get_fill();

 		assert(blocks[block_index].assert_consistency());

 		uint64_t md_after = blocks[block_index].md[0];

 		if (!found){

 			assert(md_before == md_after);

 			

 		} else {

 			assert(new_fill >= 0);

 			assert(old_fill-1 == new_fill);
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

__device__ uint64_t vqf::hash_key(uint64_t key){


	key = MurmurHash64A(((void *)&key), sizeof(key), seed) % ((num_blocks * VIRTUAL_BUCKETS) << TAG_BITS);

	return key;


}

__global__ void hash_all(vqf* my_vqf, uint64_t* vals, uint64_t* hashes, uint64_t nvals) {
	
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

__global__ void set_buffers_binary(vqf * my_vqf, uint64_t num_keys, uint64_t * keys){

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

__global__ void set_buffer_lens(vqf* my_vqf, uint64_t num_keys, uint64_t * keys){


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

__host__ uint64_t vqf::get_num_buffers(){

	uint64_t internal_num_blocks;

	cudaMemcpy(&internal_num_blocks, (uint64_t * ) this, sizeof(uint64_t), cudaMemcpyDeviceToHost);

 	cudaDeviceSynchronize();

 	return internal_num_blocks;
}


//have the VQF sort the input dataset and attach the buffers to the data

__host__ void vqf::attach_buffers(uint64_t * vals, uint64_t nvals){



	hash_all<<<(nvals - 1)/1024 + 1, 1024>>>(this, vals, vals, nvals);


	thrust::sort(thrust::device, vals, vals+nvals);




	uint64_t internal_num_blocks = get_num_buffers();
	


 	set_buffers_binary<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, nvals, vals);

 	set_buffer_lens<<<(internal_num_blocks - 1)/1024 +1, 1024>>>(this, nvals, vals);


}


__global__ void vqf_block_setup(vqf * vqf){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= vqf->num_blocks) return;

	vqf->blocks[tid].setup();

}

__host__ vqf * build_vqf(uint64_t nitems){


	#if DEBUG_ASSERTS

	printf("Debug correctness checks on. These will affect performance.\n");

	#endif

	//this seems weird but whatever
	uint64_t num_blocks = (nitems -1)/SLOTS_PER_BLOCK + 1;


	printf("Bytes used: %llu for %llu blocks.\n", num_blocks*sizeof(vqf_block),  num_blocks);


	vqf * host_vqf;

	vqf * dev_vqf;

	vqf_block * blocks;

	cudaMallocHost((void ** )& host_vqf, sizeof(vqf));

	cudaMalloc((void ** )& dev_vqf, sizeof(vqf));	

	//init host
	host_vqf->num_blocks = num_blocks;

	//allocate blocks
	cudaMalloc((void **)&blocks, num_blocks*sizeof(vqf_block));

	cudaMemset(blocks, 0, num_blocks*sizeof(vqf_block));

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


	cudaMemcpy(dev_vqf, host_vqf, sizeof(vqf), cudaMemcpyHostToDevice);

	cudaFreeHost(host_vqf);

	vqf_block_setup<<<(num_blocks - 1)/64 + 1, 64>>>(dev_vqf);
	cudaDeviceSynchronize();

	return dev_vqf;


}


__device__ uint64_t vqf::get_bucket_from_hash(uint64_t hash){

	return ((hash >> TAG_BITS) % (num_blocks * VIRTUAL_BUCKETS)) / VIRTUAL_BUCKETS;
}

#endif

