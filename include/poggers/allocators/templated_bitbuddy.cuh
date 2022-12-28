#ifndef POGGERS_TEMPLATE_BITBUDDY
#define POGGERS_TEMPLATE_BITBUDDY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/uint64_bitarray.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>


namespace cg = cooperative_groups;


#define LEVEL_CUTOF 0

#define PROG_CUTOFF 3


//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 


//compress index into 32 bit index
__host__ __device__ static int shrink_index(int index){
	if (index == -1) return index;

	return index/2;
}



template <int depth, uint64_t size_in_bytes>
struct templated_bitbuddy{


	using my_type = templated_bitbuddy<depth, size_in_bytes>;

	enum { size = size_in_bytes/32 };

	using child_type = templated_bitbuddy<depth-1, size>;

	static_assert(size > 0);
	
	
	uint64_t_bitarr mask;

	child_type children[32];


	static __host__ my_type * generate_on_device(){



		my_type * dev_version;

		cudaMalloc((void **)& dev_version, sizeof(my_type));

		cudaMemset(dev_version, ~0U, sizeof(my_type));

		return dev_version;



	}

	static __host__ void free_on_device(my_type * dev_version){


		cudaFree(dev_version);

	}


	__host__ __device__ bool valid_for_alloc(uint64_t ext_size){

		return (size >= ext_size && ext_size >= child_type::size);

	}


	__device__ uint64_t malloc_at_level(){

		while (true){

			int index = shrink_index(mask.get_random_active_bit_full());

			if (index == -1) return (~0ULL);

			uint64_t old = mask.unset_both_atomic(index);



			if (__popcll(old & READ_BOTH(index)) == 2){

				return index * size;

			} else {

				mask.reset_both_atomic(old, index);

			}


		}

	}

	__device__ uint64_t malloc_child(uint64_t bytes_needed){

		while (true){

			int index = shrink_index(mask.get_random_active_bit_control());

			if (index == -1) return (~0ULL);

			if (mask.unset_lock_bit_atomic(index) & SET_SECOND_BIT(index)){
				//valid

				uint64_t offset = children[index].malloc_offset(bytes_needed);

				if (offset == (~0ULL)){
					mask.unset_control_bit_atomic(index);
					continue;
				}

				return index*size + offset;
			}

		}

	}

	__device__ uint64_t malloc_offset(uint64_t bytes_needed){


		uint64_t offset;

		if (valid_for_alloc(bytes_needed)){

			offset = malloc_at_level();

		} else {

			offset = malloc_child(bytes_needed);

		}


		return offset;



	}


	__device__ bool free_at_level(uint64_t offset){


		if (__popcll(mask.set_both_atomic(offset) & READ_BOTH(offset)) == 0){

			return true;
		}

		return false;

	}

	__device__ bool free(uint64_t offset){

		uint64_t local_offset = offset / size;

		assert(local_offset < 32);

		if (children[local_offset].free(offset % size)){

			mask.set_control_bit_atomic(local_offset);
			return true;

		}

		return free_at_level(local_offset);

	}





};


template <uint64_t size_in_bytes>
struct  templated_bitbuddy<0, size_in_bytes> {

	using my_type = templated_bitbuddy<0, size_in_bytes>;

	enum {size = size_in_bytes};

	uint64_t_bitarr mask;


	__device__ uint64_t malloc_offset(uint64_t bytes_needed){

		return malloc_at_level();
	}


	__device__ uint64_t malloc_at_level(){


		while (true){

			int index = shrink_index(mask.get_random_active_bit_full());

			if (index == -1) return (~0ULL);

			if (__popcll(mask.unset_both_atomic(index) & READ_BOTH(index)) == 2){

				return index;

			}


		}


	}


	//returns true if entirely full
	__device__ bool free_at_level(uint64_t offset){


		if (__popcll(mask.set_both_atomic(offset) & READ_BOTH(offset)) == 0){

			return true;
		}

		return false;

	}

	__device__ bool free(uint64_t offset){

		return free_at_level(offset);

	}

	__host__ __device__ bool valid_for_alloc(uint64_t size){
		return true;
	}



	static __host__ my_type * generate_on_device(){



		my_type * dev_version;

		cudaMalloc((void **)& dev_version, sizeof(my_type));

		cudaMemset(dev_version, ~0U, sizeof(my_type));

		return dev_version;



	}

	static __host__ void free_on_device(my_type * dev_version){


		cudaFree(dev_version);

	}

};




}

}


#endif //GPU_BLOCK_