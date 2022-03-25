#ifndef TEMPLATED_VQF_H 
#define TEMPLATED_VQF_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/metadata.cuh"
#include "include/key_val_pair.cuh"
#include "include/const_size_templated_block.cuh"
#include "include/hashutil.cuh"
#include "include/templated_sorting_funcs.cuh"
#include <stdio.h>
#include <assert.h>

//thrust stuff
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>


//counters are now external to allow them to permanently reside in the l1 cache.
//this should improve performance and allow for different loading schemes
//that are less reliant on the initial load.

//these are templated on just one thing
//key_value_pairs

// template <typename Tag_type>
// __device__ bool assert_sorted(Tag_type * tags, int nitems){


// 	if (nitems < 1) return true;

// 	Tag_type smallest = tags[0];

// 	for (int i=1; i< nitems; i++){

// 		if (tags[i] < smallest) return false;

// 		smallest = tags[i];
// 	}

// 	return true;

// }


//cuda templated globals

template <typename Filter, typename Key_type>
__global__ void hash_all_key_purge(Filter * my_vqf, uint64_t * vals, Key_type * keys, uint64_t nvals){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = vals[tid];

	key = my_vqf->get_bucket_from_key(key);

	uint64_t new_key = my_vqf->get_reference_from_bucket(key) | keys[tid].get_key();

	#if DEBUG_ASSERTS

	assert(my_vqf->get_bucket_from_reference(new_key) == key);
	#endif

	//buckets are now sortable!
	vals[tid] = new_key;

}

template <typename Filter, typename Key_type>
__global__ void set_buffers_binary(Filter * my_vqf, uint64_t * references, Key_type * keys, uint64_t nvals){


		// #if DEBUG_ASSERTS

		// assert(assert_sorted(keys, nvals));

		// #endif


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
		uint64_t upper = nvals ;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			//((keys[index] >> TAG_BITS)
			uint64_t bucket = my_vqf->get_bucket_from_reference(references[index]);


			if (index != 0)
			uint64_t old_bucket = my_vqf->get_bucket_from_reference(references[index-1]);

			if (bucket < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

				//(get_bucket_from_reference(references[index-1])
				//(keys[index-1] >> TAG_BITS)

			} else if (my_vqf->get_bucket_from_reference(references[index-1]) < boundary) {

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


		my_vqf->buffers[idx] = keys + index;
		


}

template <typename Filter, typename Key_type>
__global__ void set_buffer_lens(Filter * my_vqf, uint64_t num_keys,  Key_type * keys){


	// #if DEBUG_ASSERTS

	// assert(assert_sorted(keys, num_keys));

	// #endif

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

template <typename Filter>
__global__ void sorted_bulk_insert_kernel(Filter * vqf, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	uint64_t teamID = tid / (BLOCK_SIZE);



	//TODO double check me
	if (teamID >= vqf->num_teams) return;


	//vqf->sorted_mini_filter_block(misses);

	//vqf->sorted_dev_insert(misses);
	vqf->persistent_dev_insert(misses);

	return;

	


}

template<typename Filter>
__global__ void bulk_sorted_query_kernel(Filter * vqf, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t teamID = tid / (BLOCK_SIZE);

	#if DEBUG_ASSERTS

	assert(teamID == blockIdx.x);

	#endif

	if (teamID >= vqf->num_teams) return;

	vqf->mini_filter_bulk_queries(hits);
}


//END OF KERNELS




// template <typename Filter, typename Key_type>
// __global__ void test_kernel(T * my_vqf){


// }


template <typename T>
struct __attribute__ ((__packed__)) thread_team_block {


	T internal_blocks[BLOCKS_PER_THREAD_BLOCK];

};



template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
struct __attribute__ ((__packed__)) templated_vqf {


	//tag bits change based on the #of bytes allocated per block

	using key_type = key_val_pair<Key, Val, Wrapper>;

	//typedef key_val_pair<Key, Val, Wrapper> key_type;

	using block_type = templated_block<key_type>;

	uint64_t num_teams;

	uint64_t num_blocks;

	uint64_t dividing_line;

	int * block_counters;

	key_type ** buffers;

	int * buffer_sizes;



	thread_team_block<block_type> * blocks;


	__host__ void attach_lossy_buffers(uint64_t * items, key_type * keys, uint64_t nitems, uint64_t ext_num_blocks){

		hash_all_key_purge<templated_vqf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(this, items, keys, nitems);

		thrust::sort_by_key(thrust::device, items, items+nitems, keys);


	

		set_buffers_binary<templated_vqf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, items, keys, nitems);

		set_buffer_lens<templated_vqf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, nitems, keys);


	}

	//use uint64_t and clip them down, get better variance across the structure
	__host__ void bulk_insert_lossy_keys(uint64_t * items, key_type * keys, uint64_t nitems, uint64_t ext_num_blocks){


		attach_lossy_buffers(items, keys, nitems, ext_num_blocks);


		//sorted_mini_filter_insert(uint64_t * misses); 


	}


	__host__ void bulk_insert(uint64_t * misses, uint64_t ext_num_teams){


				sorted_bulk_insert_kernel<templated_vqf<Key, Val, Wrapper>><<<ext_num_teams, BLOCK_SIZE>>>(this, misses);


	}




	//device functions
	__device__ uint64_t get_bucket_from_key(uint64_t key){

		key = MurmurHash64A(((void *)&key), sizeof(key), 42) % (num_blocks);

		return key;

	}

	__device__ uint64_t get_alt_bucket_from_key(key_type key, uint64_t bucket){

		uint64_t new_hash = MurmurHash64A((void *)&key.get_key(), sizeof(Key), 999);

		uint64_t new_bucket =  MurmurHash64A(((void *)&bucket), sizeof(bucket), 444);

		return new_hash & new_bucket; 
	}

		//device functions
	__device__ uint64_t get_bucket_from_reference(uint64_t key){

		
		return key >> (8ULL *sizeof(Key));

	}

	__device__ uint64_t get_reference_from_bucket(uint64_t key){

		
		return key << (8ULL *sizeof(Key));

	}

	__device__ void load_local_blocks(thread_team_block<block_type> * primary_block_ptr, int * local_counters, uint64_t blockID, int warpID, int threadID){

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			primary_block_ptr->internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];

			local_counters[i] = block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i];


		}
	}


	__device__ void unload_local_blocks(thread_team_block<block_type> * primary_block_ptr, int * local_counters, uint64_t blockID, int warpID, int threadID){

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			blocks[blockIdx.x].internal_blocks[i] = primary_block_ptr->internal_blocks[i];

			block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = local_counters[i];

		}
	}

	//this version loads and unloads the blocks and performs all ops in shared mem
	__device__ void persistent_dev_insert(uint64_t * misses){

		__shared__ thread_team_block<block_type> primary_block;

		//__shared__ thread_team_block<block_type> alt_storage_block;

		__shared__ int local_counters[BLOCKS_PER_THREAD_BLOCK];  

		__shared__ int buffer_offsets[BLOCKS_PER_THREAD_BLOCK];

		//__shared__ int secondary_buffer_counters[BLOCKS_PER_THREAD_BLOCK];


		thread_team_block<block_type> * primary_block_ptr = &primary_block;

		//thread_team_block<block_type> * alt_storage_block_ptr = &alt_storage_block;

		//load blocks

		uint64_t blockID = blockIdx.x;

		int warpID = threadIdx.x / 32;

		int threadID = threadIdx.x % 32;

		// for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		// 	primary_block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];

		// 	local_counters[i] = block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i];

		// }

		load_local_blocks(primary_block_ptr, &local_counters[0], blockID, warpID, threadID);


		//get counters for new_items

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		buffer_get_primary_count(primary_block_ptr, (int *)& buffer_offsets, blockID, i ,warpID, threadID);

		primary_block_ptr->internal_blocks[i].dump_multiple_items(buffers[blockID*BLOCKS_PER_THREAD_BLOCK+i], buffer_offsets[i], threadID);

		local_counters[i] += buffer_offsets[i];

		}
	
		dump_all_buffers_into_local_block(primary_block_ptr, &buffer_offsets[0], &local_counters[0], blockID, warpID, threadID, misses);




		unload_local_blocks(primary_block_ptr, &local_counters[0], blockID, warpID, threadID);

		//unload from primary ptr
		// for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		// 	blocks[blockIdx.x].internal_blocks[i] = primary_block_ptr->internal_blocks[i];

		// 	block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = local_counters[i];
		// }


	}




	__device__ void buffer_get_primary_count(thread_team_block<block_type> * local_blocks, int * counters, uint64_t blockID, int warpID, int block_warpID, int threadID){


		#if DEBUG_ASSERTS

		assert(blockID < num_teams);

		assert((blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);
		#endif




		uint64_t global_buffer = blockID * BLOCKS_PER_THREAD_BLOCK + warpID;



		int count = block_type::fill_cutoff() - block_counters[global_buffer];


		int buf_size = buffer_sizes[global_buffer];

		if (buf_size < count) count = buf_size;

		if (count < 0) count = 0;

		#if DEBUG_ASSERTS

		assert(count < block_type::max_size());

		#endif

		counters[warpID] = count;

	}



__device__ void dump_item_into_block(uint64_t reference, key_type hash, int threadID, uint64_t * misses){


		uint64_t bucket = get_bucket_from_key(reference);

		#if DEBUG_ASSERTS

		assert(bucket < num_blocks);
		#endif

		uint64_t blockID = bucket / BLOCKS_PER_THREAD_BLOCK;

		int i = bucket % BLOCKS_PER_THREAD_BLOCK;

		int alt_bucket = get_alt_bucket_from_key(hash, bucket) % BLOCKS_PER_THREAD_BLOCK;

		if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;

		//i and alt bucket set, now do atomic ops

		uint64_t global_alt_bucket = alt_bucket+blockID*BLOCKS_PER_THREAD_BLOCK;


		#if DEBUG_ASSERTS

		assert(blockID*BLOCKS_PER_THREAD_BLOCK + i == bucket);
		assert(blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket == global_alt_bucket);

		#endif


		bool main_smaller = (threadID == 0 && atomicCAS(&block_counters[bucket], 0, 0) < atomicCAS(&block_counters[global_alt_bucket],0,0));

		main_smaller = __shfl_sync(0xffffffff, main_smaller, 0);

		if (main_smaller){



			int reserved_slot = 0;
			if (threadID ==0){

				reserved_slot = atomicAdd(&block_counters[bucket], 1);

			}
			bool main_add = (reserved_slot < block_type::max_size());
			main_add = __shfl_sync(0xffffffff, main_add, 0);

			//These adds aren't undone on failure as no one else can succeed.
			if (main_add){

				//slot - offset = fill+#writes - this is guaranteed to be a free slot

				blocks[blockID].internal_blocks[i].dump_single_item_reserved_slot(hash, reserved_slot, threadID);
				//local_blocks->internal_blocks[i].tags[slot] = hash;

				#if DEBUG_ASSERTS

				bool found;

				blocks[blockID].internal_blocks[i].query_single_item(hash, threadID, found);
				assert(found);

				#endif

			} else {

				//atomicSub(&block_counters[i],1);

				int reserved_slot = 0;
				if (threadID ==0){

					reserved_slot = atomicAdd(&block_counters[global_alt_bucket], 1);
				}
				bool alt_add = (reserved_slot < block_type::max_size());

				alt_add = __shfl_sync(0xffffffff, alt_add, 0);


				//atomicadd fails, try alternate spot
				//slot = atomicAdd(&block_counters[alt_bucket], 1);

				if (alt_add){

				


					//local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;
					blocks[blockID].internal_blocks[alt_bucket].dump_single_item_reserved_slot(hash, reserved_slot, threadID);

					#if DEBUG_ASSERTS

					bool found;

					blocks[blockID].internal_blocks[alt_bucket].query_single_item(hash, threadID, found);
					assert(found);

					#endif



				} else {

					//atomicSub(&block_counters[alt_bucket],1);
					if (threadID == 0){

						atomicAdd((unsigned long long int *) misses, 1ULL);

					}

				}



			}


		} else {

			//alt < main slot
			int reserved_slot = 0;
			if (threadID ==0){

				reserved_slot = atomicAdd(&block_counters[global_alt_bucket], 1);
			}
			bool alt_add = (reserved_slot < block_type::max_size());

			alt_add = __shfl_sync(0xffffffff, alt_add, 0);

			if (alt_add){


				//temp_tags[alt_bucket*block_type::max_size()+slot] = hash & 0xff;
				blocks[blockID].internal_blocks[alt_bucket].dump_single_item_reserved_slot(hash, reserved_slot, threadID);

				//you have a write reserved, attempt again?

				#if DEBUG_ASSERTS

				bool found;

				blocks[blockID].internal_blocks[alt_bucket].query_single_item(hash, threadID, found);
				
				if (!found){
					assert(found);
				}
				

				#endif
			

			} else {

				//atomicSub(&block_counters[alt_bucket], 1); 

				//primary insert failed, attempt secondary
				int reserved_slot = 0;

				if (threadID ==0){

					reserved_slot = atomicAdd(&block_counters[bucket], 1);
				}
				bool main_add = (reserved_slot < block_type::max_size());
				
				main_add = __shfl_sync(0xffffffff, main_add, 0);

				if (main_add){


					//temp_tags[i*block_type::max_size()+slot] = hash & 0xff;

					blocks[blockID].internal_blocks[i].dump_single_item_reserved_slot(hash, reserved_slot, threadID);

					//local_blocks->internal_blocks[i].tags[slot] = hash;

					#if DEBUG_ASSERTS

					bool found;

					blocks[blockID].internal_blocks[i].query_single_item(hash, threadID, found);
					assert(found);

					#endif

				} else {


					//atomicSub(&counters[alt_bucket], 1);
					if (threadID == 0){
						atomicAdd((unsigned long long int *) misses, 1ULL);
					}
					


					}

				}




		}


	//end of loop

}


__device__ bool query_single_item(uint64_t reference, key_type hash, int threadID){


		uint64_t bucket = get_bucket_from_key(reference);

		#if DEBUG_ASSERTS

		assert(bucket < num_blocks);
		#endif

		uint64_t blockID = bucket / BLOCKS_PER_THREAD_BLOCK;

		int i = bucket % BLOCKS_PER_THREAD_BLOCK;

		int alt_bucket = get_alt_bucket_from_key(hash, bucket) % BLOCKS_PER_THREAD_BLOCK;

		if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;

		//i and alt bucket set, now do atomic ops

		uint64_t global_alt_bucket = alt_bucket+blockID*BLOCKS_PER_THREAD_BLOCK;

		#if DEBUG_ASSERTS

		assert(blockID*BLOCKS_PER_THREAD_BLOCK + i == bucket);
		assert(blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket == global_alt_bucket);

		#endif

		bool found = false;

		blocks[blockID].internal_blocks[i].query_single_item(hash, threadID, found);


		if (!found) blocks[blockID].internal_blocks[alt_bucket].query_single_item(hash, threadID, found);

		// if (! found){
		// 	assert(found);
		// }

		return found;





}

__device__ void dump_all_buffers_into_local_block(thread_team_block<block_type> * local_blocks, int * offsets, int * counters, uint64_t blockID, int warpID, int threadID, uint64_t * misses){





		#if DEBUG_ASSERTS

		// for (int i = 0; i < BLOCKS_PER_THREAD_BLOCK; i++){

		// 	assert(counters[i] <= block_type::max_size());
		// }

		__syncthreads();

		#endif


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


			//for each item in parallel, we check the global counters to determine which hash is submitted
			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;

			int remaining = buffer_sizes[global_buffer] - offsets[i];


			//threads must work together on these ops
			for (int j = 0; j < remaining; j+=1){


				key_type hash = buffers[global_buffer][j+offsets[i]];

				//uint64_t  = get_alt_hash(hash, global_buffer);

				int alt_bucket = get_alt_bucket_from_key(hash, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

				if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


				#if DEBUG_ASSERTS

				assert(j + offsets[i] < buffer_sizes[global_buffer]);

				assert(alt_bucket < BLOCKS_PER_THREAD_BLOCK);
				assert(i < BLOCKS_PER_THREAD_BLOCK);
				assert(alt_bucket != i);


				#endif

				//replace with faster atomic

				//4 atomics, 3 syncs required
				//all handled by threadId 0 as the master
				
				bool main_smaller = (threadID == 0 && atomicCAS(&counters[i], 0, 0) < atomicCAS(&counters[alt_bucket],0,0));

				main_smaller = __shfl_sync(0xffffffff, main_smaller, 0);

				if (main_smaller){



					int reserved_slot = 0;
					if (threadID ==0){

						reserved_slot = atomicAdd(&counters[i], 1);

					}
					bool main_add = (reserved_slot < block_type::max_size());
					main_add = __shfl_sync(0xffffffff, main_add, 0);

					//These adds aren't undone on failure as no one else can succeed.
					if (main_add){

						//slot - offset = fill+#writes - this is guaranteed to be a free slot

						local_blocks->internal_blocks[i].dump_single_item_reserved_slot(hash, reserved_slot, threadID);
						//local_blocks->internal_blocks[i].tags[slot] = hash;

						#if DEBUG_ASSERTS

						bool found;

						local_blocks->internal_blocks[i].query_single_item(hash, threadID, found);
						assert(found);

						#endif

					} else {

						//atomicSub(&counters[i],1);

						int reserved_slot = 0;
						if (threadID ==0){

							reserved_slot = atomicAdd(&counters[alt_bucket], 1);
						}
						bool alt_add = (reserved_slot < block_type::max_size());

						alt_add = __shfl_sync(0xffffffff, alt_add, 0);


						//atomicadd fails, try alternate spot
						//slot = atomicAdd(&counters[alt_bucket], 1);

						if (alt_add){

						


							//local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;
							local_blocks->internal_blocks[alt_bucket].dump_single_item_reserved_slot(hash, reserved_slot, threadID);

							#if DEBUG_ASSERTS

							bool found;

							local_blocks->internal_blocks[alt_bucket].query_single_item(hash, threadID, found);
							assert(found);

							#endif



						} else {

							//atomicSub(&counters[alt_bucket],1);
							if (threadID == 0){

								atomicAdd((unsigned long long int *) misses, 1ULL);

							}

						}



					}


				} else {

					//alt < main slot
					int reserved_slot = 0;
					if (threadID ==0){

						reserved_slot = atomicAdd(&counters[alt_bucket], 1);
					}
					bool alt_add = (reserved_slot < block_type::max_size());

					alt_add = __shfl_sync(0xffffffff, alt_add, 0);

					if (alt_add){


						//temp_tags[alt_bucket*block_type::max_size()+slot] = hash & 0xff;
						local_blocks->internal_blocks[alt_bucket].dump_single_item_reserved_slot(hash, reserved_slot, threadID);

						//you have a write reserved, attempt again?

						#if DEBUG_ASSERTS

						bool found;

						local_blocks->internal_blocks[alt_bucket].query_single_item(hash, threadID, found);
						
						if (!found){
							assert(found);
						}
						

						#endif
					

					} else {

						//atomicSub(&counters[alt_bucket], 1); 

						//primary insert failed, attempt secondary
						int reserved_slot = 0;

						if (threadID ==0){

							reserved_slot = atomicAdd(&counters[i], 1);
						}
						bool main_add = (reserved_slot < block_type::max_size());
						
						main_add = __shfl_sync(0xffffffff, main_add, 0);

						if (main_add){


							//temp_tags[i*block_type::max_size()+slot] = hash & 0xff;

							local_blocks->internal_blocks[i].dump_single_item_reserved_slot(hash, reserved_slot, threadID);

							//local_blocks->internal_blocks[i].tags[slot] = hash;

							#if DEBUG_ASSERTS

							bool found;

							local_blocks->internal_blocks[i].query_single_item(hash, threadID, found);
							assert(found);

							#endif

						} else {


							//atomicSub(&counters[alt_bucket], 1);
							if (threadID == 0){
								atomicAdd((unsigned long long int *) misses, 1ULL);
							}
							


							}

						}




				}


			}

		}


		//end of for loop

		__syncthreads();


	}



	//Queries
	__host__ void bulk_query(bool * hits, uint64_t ext_num_teams){


		bulk_sorted_query_kernel<templated_vqf<Key, Val, Wrapper>><<<ext_num_teams, BLOCK_SIZE>>>(this, hits);



	}




	__device__ bool mini_filter_bulk_queries(bool * hits){

		__shared__ thread_team_block<block_type> block;

		//uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

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

			for (int j = 0; j < buffer_sizes[global_buffer]; j++){

				bool temp_hits;
				block.internal_blocks[i].query_single_item(buffers[global_buffer][j], threadID, temp_hits);
				hits_ptr[j] = temp_hits;

			}

			//block.internal_blocks[i].sorted_bulk_query(block_counters[global_buffer], threadID, buffers[global_buffer], hits_ptr, buffer_sizes[global_buffer]);

		}


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;


			for (int j = 0; j < buffer_sizes[global_buffer]; j+=1){

				if (!hits_ptr[j]){

					key_type item = buffers[global_buffer][j];

					int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

					if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

					bool temp_hits;
					block.internal_blocks[alt_bucket].query_single_item(item, threadID, temp_hits);
					hits_ptr[j] = temp_hits;

					// if (!hits_ptr[j]){
					// 	assert(hits_ptr[j]);
					// }
				}

			}
		}

		__syncthreads();

		return true;

	}


	__host__ uint64_t get_num_blocks(){


		templated_vqf<Key, Val, Wrapper> * host_vqf;


		cudaMallocHost((void **)& host_vqf, sizeof(templated_vqf<Key, Val, Wrapper>));

		cudaMemcpy(host_vqf, this, sizeof(templated_vqf<Key, Val, Wrapper>), cudaMemcpyDeviceToHost);

		uint64_t blocks_val = host_vqf->num_blocks;

		cudaFreeHost(host_vqf);

		return blocks_val;


	}


	__host__ uint64_t get_num_teams(){


		templated_vqf<Key, Val, Wrapper> * host_vqf;


		cudaMallocHost((void **)& host_vqf, sizeof(templated_vqf<Key, Val, Wrapper>));

		cudaMemcpy(host_vqf, this, sizeof(templated_vqf<Key, Val, Wrapper>), cudaMemcpyDeviceToHost);

		uint64_t teams_val = host_vqf->num_teams;

		cudaFreeHost(host_vqf);


		return teams_val;


	}




	//these bad boys are exact!
	//__host__ bulk_insert(key_type * keys);
	

};




template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
__host__ void free_vqf(templated_vqf<Key, Val, Wrapper> * vqf){


	templated_vqf<Key, Val, Wrapper> * host_vqf;


	cudaMallocHost((void **)& host_vqf, sizeof(templated_vqf<Key, Val, Wrapper>));

	cudaMemcpy(host_vqf, vqf, sizeof(templated_vqf<Key, Val, Wrapper>), cudaMemcpyDeviceToHost);

	cudaFree(vqf);

	cudaFree(host_vqf->blocks);

	cudaFree(host_vqf->block_counters);

	cudaFree(host_vqf->buffers);
	cudaFree(host_vqf->buffer_sizes);

	cudaFreeHost(host_vqf);




}


template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
__host__ templated_vqf<Key, Val, Wrapper> * build_vqf(uint64_t nitems){


	using key_type = key_val_pair<Key, Val, Wrapper>;

	using block_type = templated_block<key_type>;

	templated_vqf<Key, Val, Wrapper> * host_vqf;

	cudaMallocHost((void **)&host_vqf, sizeof(templated_vqf<Key,Val,Wrapper>));

	uint64_t num_teams = (nitems - 1)/(BLOCKS_PER_THREAD_BLOCK*block_type::max_size()) + 1;

	uint64_t num_blocks = num_teams*BLOCKS_PER_THREAD_BLOCK;

	printf("VQF hash hash %llu thread_team_blocks of %d blocks, total %llu blocks\n", num_teams, BLOCKS_PER_THREAD_BLOCK, num_blocks);
	printf("Each block is %llu items of size %d, total size %d\n", block_type::max_size(), sizeof(key_type), block_type::max_size()*sizeof(key_type));


	host_vqf->num_teams = num_teams;
	host_vqf->num_blocks = num_blocks;


	int * counters;

	cudaMalloc((void **)&counters, num_blocks*sizeof(int));

	cudaMemset(counters, 0, num_blocks*sizeof(int));

	host_vqf->block_counters = counters;


	thread_team_block<block_type> * blocks;

	cudaMalloc((void **)& blocks, num_teams*sizeof(thread_team_block<block_type>));


	host_vqf->blocks = blocks;

	//this should 
	host_vqf->dividing_line = (1ULL << (8*sizeof(Key)-5));
	//buffers

	printf("dividing_line: %llu\n", host_vqf->dividing_line);

	key_type ** buffers;

	cudaMalloc((void **)&buffers, num_blocks*sizeof(key_type **));

	host_vqf->buffers = buffers;

	int * buffer_sizes;

	cudaMalloc((void **)&buffer_sizes, num_blocks*sizeof(int));

	host_vqf->buffer_sizes = buffer_sizes;


	templated_vqf<Key, Val, Wrapper> * dev_vqf;


	cudaMalloc((void **)& dev_vqf, sizeof(templated_vqf<Key, Val, Wrapper>));

	cudaMemcpy(dev_vqf, host_vqf, sizeof(templated_vqf<Key, Val, Wrapper>), cudaMemcpyHostToDevice);

	cudaFreeHost(host_vqf);



	return dev_vqf;

}


#endif //GPU_BLOCK_