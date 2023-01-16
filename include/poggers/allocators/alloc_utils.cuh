#ifndef POGGERS_ALLOC_UTILS
#define POGGERS_ALLOC_UTILS


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include "stdio.h"
#include "assert.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


//a pointer list managing a set section of device memory
namespace poggers {


namespace utils { 


__device__ inline uint ldca(const uint *p) {
	uint res;
	asm volatile("ld.global.ca.u32 %0, [%1];": "=r"(res) : "l"(p));
	return res;
}

__device__ inline uint64_t ldca(const uint64_t *p) {
	uint64_t res;
	asm volatile("ld.global.ca.u64 %0, [%1];": "=l"(res) : "l"(p));
	return res;

	//return atomicOr((unsigned long long int *)p, 0ULL);
}  

__device__ inline void *ldca(void * const *p) {
	void *res;
	asm volatile("ld.global.ca.u64 %0, [%1];": "=l"(res) : "l"(p));
	return res;
}  

/** prefetches into L1 cache */
__device__ inline void prefetch_l1(const void *p) {
	asm("prefetch.global.L1 [%0];": :"l"(p));
}

/** prefetches into L2 cache */
__device__ inline void prefetch_l2(const void *p) {
	asm("prefetch.global.L2 [%0];": :"l"(p));
}

__device__ uint get_smid() {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}


__host__ int get_num_streaming_multiprocessors(int which_device){

	cudaDeviceProp prop;
    	cudaGetDeviceProperties(&prop, which_device);
     int mp = prop.multiProcessorCount;

     return mp;

}

// __device__ uint64_t reduce_less(cg::coalesced_threads active_threads, uint64_t val){

// 	int i = 1;



// }


}

}


#endif //GPU_BLOCK_