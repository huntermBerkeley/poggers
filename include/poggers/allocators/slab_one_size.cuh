#ifndef SLAB_ONE_SIZE
#define SLAB_ONE_SIZE


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/offset_slab.cuh>
#include <poggers/allocators/one_size_allocator.cuh>
#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#define SLAB_ONE_SIZE_MAX_ATTEMPTS 10

namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 

template <int extra_blocks>
struct one_size_slab_allocator {


	using my_type = one_size_slab_allocator<extra_blocks>;

	//doesn't seem necessary tbh
	//uint64_t offset_size;
	uint64_t offset_size;
	one_size_allocator * block_allocator;
	//one_size_allocator * mem_alloc;
	char * extra_memory;

	smid_pinned_container<extra_blocks> * malloc_containers;

	pinned_storage * storage_containers;

	//add hash table type here.
	//map hashes to bytes?


	static __host__ my_type * generate_on_device(uint64_t num_allocs, uint64_t ext_size){


	my_type * host_version;

	cudaMallocHost((void **)&host_version, sizeof(my_type));

	host_version->offset_size = ext_size;

	uint64_t num_pinned_blocks = (num_allocs-1)/4096+1;

	host_version->block_allocator = one_size_allocator::generate_on_device(num_pinned_blocks, sizeof(offset_alloc_bitarr), 17);

    //host_version->mem_allocator = one_size_allocator::generate_on_device(num_pinned_blocks, 4096*ext_size, 1324);

	char * host_ptr_ext_mem;
	cudaMalloc((void **)&host_ptr_ext_mem, num_pinned_blocks*ext_size*4096);

	host_version->extra_memory = host_ptr_ext_mem;

 	host_version->malloc_containers = smid_pinned_container<extra_blocks>::generate_on_device(host_version->block_allocator, 4096);

 	host_version->storage_containers = pinned_storage::generate_on_device();


 	my_type * dev_version;

 	cudaMalloc((void **)&dev_version, sizeof(my_type));

 	cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

 	cudaFreeHost(host_version);

 	cudaDeviceSynchronize();

 	return dev_version;


	}


	static __host__ void free_on_device(my_type * dev_version){

		my_type * host_version;
		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		one_size_allocator::free_on_device(host_version->block_allocator);
		//one_size_allocator::free_on_device(host_version->mem_allocator);

		cudaFree(host_version->extra_memory);

		smid_pinned_container<extra_blocks>::free_on_device(host_version->malloc_containers);

		pinned_storage::free_on_device(host_version->storage_containers);

		cudaFree(dev_version);

		cudaFreeHost(host_version);

		return;


	}

	__device__ void * malloc(){

		//__shared__ warp_lock team_lock;

		smid_pinned_storage<extra_blocks> * my_storage = malloc_containers->get_pinned_storage();

   		offset_storage_bitmap * my_storage_bitmap = storage_containers->get_pinned_storage();

   		int num_attempts = 0;

   		while (num_attempts < SLAB_ONE_SIZE_MAX_ATTEMPTS){

   			//auto team = cg::coalesced_threads();


   			offset_alloc_bitarr * bitarr = my_storage->get_primary();

   			if (bitarr == nullptr){
   				//team.sync();
   				continue;
   			}


   			uint64_t allocation;

   			bool alloced = alloc_with_locks(allocation, bitarr, my_storage_bitmap);

   			if (!alloced){
   				int result = my_storage->pivot_primary(bitarr);



   				if (result != -1){

   					//malloc and replace pivot slab

   					{
   						uint64_t slab_offset = block_allocator->get_offset();

   						offset_alloc_bitarr * slab = (offset_alloc_bitarr *) block_allocator->get_mem_from_offset(slab_offset);

   						slab->init();

   						slab_offset = slab_offset*offset_size;

   						slab->attach_allocation(slab_offset);

   						slab->mark_pinned();

   						my_storage->attach_new_buffer(result, slab);


   					}

   				}


   			} else {

   				return (void *) (extra_memory + allocation*offset_size);


   			}

   			num_attempts+=1;


   		}


	}

	__device__ uint64_t get_offset_from_ptr(void * ext_ptr){

		//first off cast to uint64_t

		uint64_t ext_as_bits = (uint64_t) ext_ptr;

		//now downshift and subtract

		ext_as_bits = ext_as_bits - (uint64_t) extra_memory;

		ext_as_bits = ext_as_bits/offset_size;

		return ext_as_bits

	}


	//in the one allocator scheme free is simplified - get the block and free
	//if the block we free to is unpinned, we can safely return the memory to the veb tree
	__device__ void free(void * ext_allocation){

		uint64_t allocation_offset = get_offset_from_ptr(ext_allocation);

		uint64_t slab_offset = allocation_offset/4096;

		offset_alloc_bitarr * slab = (offset_alloc_bitarr * )  block_allocator->get_mem_from_offset(slab_offset);

		if (slab->free_allocation_v2(allocation_offset)){

			//slab may be available for free - need to check pin status.
			if (slab->atomic_check_unpinned()){

				//slabs that are marked unpinned cannot be reattached - therefore, this read succeeding guarantees correctness.
				block_allocator->free_offset(slab_offset);

			}

		}

	}




};


}

}


#endif //GPU_BLOCK_