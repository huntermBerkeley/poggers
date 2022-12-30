#ifndef POGGERS_BITBUDDY
#define POGGERS_BITBUDDY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/templated_bitbuddy.cuh>
#include <poggers/allocators/uint64_bitarray.cuh>
#include <poggers/allocators/slab.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>


namespace cg = cooperative_groups;



//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 


	//the final outcome of our experiments: one unified allocator for GPU
	//bitbuddy supplies high level allocations, 
	struct merged_allocator {






	};


}

}


#endif //GPU_BLOCK_