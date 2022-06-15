#ifndef REP_HELPERS 
#define REP_HELPERS


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace helpers {


template <typename T>
__device__ __inline__ bool get_evenness_fast(T item){
	return item & 1;
}


//These, of course, are atomics
//don't call these on stack variables

template<typename T>
__device__ __inline__ bool typed_atomic_write(T * backing, T item, T replace){


	//atomic CAS first bit

	//this should break, like you'd expect it to
	//TODO come back and make this convert to uint64_t for CAS
	//you can't CAS anything smaller than 16 bits so I'm not going to attempt that

	//printf("I am being accessed\n");

	static_assert(sizeof(T) > 8);

	uint64_t uint_item = ((uint64_t *) &item)[0];

	uint64_t uint_replace = ((uint64_t *) &replace)[0];

	if (typed_atomic_write<uint64_t>((uint64_t *) backing, uint_item, uint_replace)){

		//succesful? - flush write
		backing[0] = replace;
		return true;

	}

	return false;
}


template<>
__device__ __inline__ bool typed_atomic_write<uint16_t>(uint16_t * backing, uint16_t item, uint16_t replace){


	return (atomicCAS((unsigned short int *) backing, (unsigned short int) item, (unsigned short int) replace) == item);

}


template<>
__device__ __inline__ bool typed_atomic_write<uint32_t>(uint32_t * backing, uint32_t item, uint32_t replace){


	return (atomicCAS((unsigned int *) backing, (unsigned int) item, (unsigned int) replace) == item);

}

template<>
__device__ __inline__ bool typed_atomic_write<uint64_t>(uint64_t * backing, uint64_t item, uint64_t replace){

	//printf("Uint64_t call being accessed\n");

	return (atomicCAS((unsigned long long int *) backing, (unsigned long long int) item, (unsigned long long int) replace) == item);

}


}

}


#endif //GPU_BLOCK_