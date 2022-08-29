#ifndef CMS
#define CMS


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/aligned_stack.cuh>
#include <poggers/allocators/sub_allocator.cuh>

#include "stdio.h"
#include "assert.h"


//include files for the hash table
//hash scheme
#include <poggers/hash_schemes/murmurhash.cuh>

//probing scheme
#include <poggers/probing_schemes/double_hashing.cuh>

//insert_scheme
#include <poggers/insert_schemes/bucket_insert.cuh>

//table type
#include <poggers/tables/base_table.cuh>

//storage containers for keys
#include <poggers/representations/dynamic_container.cuh>
#include <poggers/representations/key_only.cuh>

//sizing type for building the table
#include <poggers/sizing/default_sizing.cuh>

//A series of inclusions for building a poggers hash table


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//CMS: The CUDA Memory Shibboleth
//CMS is a drop-in replacement for cudaMalloc and CudaFree.
//Before your kernels start, initialize a handler with
//shibboleth * manager = poggers::allocators::shibboleth::init() or init_managed() for host-device unified memory.
// The amount of memory specified at construction is all that's available to the manager,
// so you can spin up multiple managers for different tasks or request all available memory!
// The memory returned is built off of cudaMalloc or cudaMallocManaged, so regular cuda API calls are fine.
// Unlike the cuda device API, however, you can safely cudaMemcpy to and from memory requested by threads!




//a pointer list managing a set section o fdevice memory

// 	const float log_of_size = std::log2()

// }

namespace poggers {


namespace allocators { 


template <typename allocator>
__global__ void allocate_stack(allocator ** stack_ptr, header * heap){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0) return;

   allocator * my_stack = allocator::init(heap);

   stack_ptr[0] = my_stack;

   return;

}

template <typename allocator>
__host__ allocator * host_allocate_sub_allocator(header * heap){

   allocator ** stack_ptr;

   cudaMallocManaged((void **)&stack_ptr, sizeof(allocator *));

   allocate_stack<allocator><<<1,1>>>(stack_ptr, heap);

   cudaDeviceSynchronize();

   allocator * to_return = stack_ptr[0];

   cudaFree(stack_ptr);

   return to_return;



}


template <std::size_t bytes_per_substack, std::size_t num_suballocators, std::size_t maximum_p2>
struct shibboleth {


	using stack_type = aligned_manager<bytes_per_substack, false>;
	using my_type = shibboleth<bytes_per_substack, num_suballocators, maximum_p2>;

	using allocator = sub_allocator<bytes_per_substack, maximum_p2>;

	using heap = header;

	using hash_table = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_container,uint64_t>::representation, 1, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

	allocator * allocators[num_suballocators];

	heap * allocated_memory;

	hash_table * free_table;




	static __host__ my_type * init_backend(std::size_t bytes_requested, bool managed){

		//this is gonna be very cheeky
		//an allocator can be hosted on its own memory!

		my_type * host_cms = (my_type * ) malloc(sizeof(my_type)); 

		//allocated_memory;

		if (managed){
			host_cms->allocated_memory = heap::init_heap_managed(bytes_requested);
		} else {
			host_cms->allocated_memory = heap::init_heap(bytes_requested);
		}


		if (host_cms->allocated_memory == nullptr){
			printf("Failed to allocate device memory!\n");
			abort();
		}

		for (int i = 0; i < num_suballocators; i++){
			host_cms->allocators[i] = host_allocate_sub_allocator<allocator>(host_cms->allocated_memory);
			if (host_cms->allocators[i] == nullptr){
				printf("Not enough space for allocator %d\n", i);
				abort();
			}
		}


		//calculate maximum number of stacks possible
		//and store internally

		//TODO:
		//move table generation to the heap
		//so this can be self contained
		//estimates but table size at .1% of the memory allocated

		uint64_t max_stacks = (bytes_requested-1)/bytes_per_substack + 1;

		//boot hash table here
		poggers::sizing::size_in_num_slots<1> slots_for_table(max_stacks*1.1);
		host_cms->free_table = hash_table::generate_on_device(&slots_for_table, 42);

		//forcing alignment to 16 bytes to guarantee cache alignment for atomics
		my_type * dev_cms = (my_type *) host_cms->allocated_memory->host_malloc_aligned(sizeof(my_type), 16, 0);

		if (dev_cms == nullptr){
			printf("Not enough space for dev ptr\n");
			abort();
		}

		cudaMemcpy(dev_cms, host_cms, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		free(host_cms);

		return dev_cms;


	}

	__host__ static my_type * init(std::size_t bytes_requested){

		return init_backend(bytes_requested, false);

	}

	__host__ static my_type * init_managed(std::size_t bytes_requested){

		return init_backend(bytes_requested, true);
		
	}


	//to free, get handle to memory and just release
	__host__ static void free_cms_allocator(my_type * dev_cms){

		my_type * host_cms = (my_type * ) malloc(sizeof(my_type));

		cudaMemcpy(host_cms, dev_cms, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		hash_table::free_on_device(host_cms->free_table);

		heap::free_heap(host_cms->allocated_memory);

		return;

	}


	__device__ allocator * get_randomish_sub_allocator(){

		poggers::hashers::murmurHasher<uint64_t,1> randomish_hasher;

		randomish_hasher.init(42);

		uint64_t very_random_number = clock64();

		return allocators[randomish_hasher.hash(very_random_number) % num_suballocators];

	}


	//malloc splits requests into two groups
	//free list mallocs and stack mallocs
	//we default to stack and only upgrade iff no stack is large enough to handle
	__device__ void * cms_malloc(uint64_t num_bytes){

		if (allocator::can_malloc(num_bytes)){

			allocator * randomish_allocator = get_randomish_sub_allocator();

			//void * ret_value = randomish_allocator->malloc_free_table<hash_table>(num_bytes, free_table, allocated_memory);
			int * test_ptr;

			//randomish_allocator->template test_load<int>(test_ptr);
			//randomish_allocator->test_load<hash_table>(free_table);

			return get_randomish_sub_allocator()->template malloc_free_table<hash_table>(num_bytes, free_table, allocated_memory);

		} else {

			return allocated_memory->malloc(num_bytes);

		}

	}

	__device__ void cms_free(void * uncasted_address){

		uint64_t home_uint = stack_type::get_home_address_uint(uncasted_address);

		bool found;

		{
			auto tile = free_table->get_my_tile();
			uint16_t temp_val = 0;
			found = free_table->query(tile, home_uint, temp_val);
		}

		if (found){
			stack_type::static_free(uncasted_address);
		} else {
			allocated_memory->free_safe(uncasted_address);
		}

	}





};


}

}


#endif //GPU_BLOCK_