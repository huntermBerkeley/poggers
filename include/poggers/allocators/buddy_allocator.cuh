#ifndef POGGERS_BUDDY_ALLOCATOR
#define POGGERS_BUDDY_ALLOCATOR


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <stdio.h>
#include "assert.h"

//a pointer list managing a set section of device memory


//include reporter for generating usage reports
#include <poggers/allocators/reporter.cuh>

// #define DEBUG_ASSERTS 1

#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//define CUTOFF_SIZE 1024
#define CUTOFF_SIZE 50

#if COUNTING_CYCLES
#include <poggers/allocators/cycle_counting.cuh>
#endif


struct buddy_internal_bitfield {
	uint64_t counter : 5;
	uint64_t pointer : 27; 

};

union buddy_unioned_bitfield {

	buddy_internal_bitfield as_bitfield;
	uint64_t as_uint;

};



namespace poggers {


namespace allocators { 


struct buddy_node {

	buddy_unioned_bitfield * next;
	buddy_unioned_bitfield * prev;

	__device__ buddy_node * get_next_atomic(){

		buddy_unioned_bitfield * ptr_to_next = (buddy_unioned_bitfield *)atomicOr((unsigned long long int *)&next, 0ULL);

		return ptr_to_next;

	}

	__device__ buddy_node * get_prev_atomic(){

		buddy_unioned_bitfield * ptr_to_prev = (buddy_unioned_bitfield *)atomicOr((unsigned long long int *)&prev, 0ULL);

		return ptr_to_prev;

	}

};

template <size_t smallest, size_t largest>
struct buddy_allocator {


};


}

}


#endif //GPU_BLOCK_