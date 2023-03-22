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


//helper_macro
//define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))



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

//for a given template family, how many chunks do they need?
template<uint64_t bytes_per_chunk>
__host__ uint64_t get_max_chunks(){

		size_t mem_total;
		size_t mem_free;
		cudaMemGetInfo  (&mem_free, &mem_total);

		return mem_total/bytes_per_chunk;

}

template<uint64_t bytes_per_chunk>
__host__ uint64_t get_max_chunks(max_bytes){

		return max_bytes/bytes_per_chunk;

}


template <typename Struct_Type>
__host__ Struct_Type * get_host_version(){

	Struct_Type * host_version;

	cudaMallocHost((void **)&host_version, sizeof(Struct_Type));

	return host_version;

}

template <typename Struct_Type>
__host__ Struct_Type * get_host_version(uint64_t num_copies){

	Struct_Type * host_version;

	cudaMallocHost((void **)&host_version, num_copies*sizeof(Struct_Type));

	return host_version;

}

template <typename Struct_Type>
__host__ Struct_Type * get_device_version(){

	Struct_Type * dev_version;

	cudaMalloc((void **)&dev_version, sizeof(Struct_Type));

	return dev_version;

}

template <typename Struct_Type>
__host__ Struct_Type * get_device_version(uint64_t num_copies){

	Struct_Type * dev_version;

	cudaMalloc((void **)&dev_version, num_copies*sizeof(Struct_Type));

	return dev_version;

}

template <typename Struct_Type>
__host__ Struct_Type * move_to_device(Struct_Type * host_version){

	dev_version * get_device_version();

	cudaMemcpy(dev_version, host_version, sizeof(Struct_Type), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	cudaFree(host_version);

	return dev_version;


}

template <typename Struct_Type>
__host__ Struct_Type * move_to_host(Struct_Type * dev_version){

	host_version * get_host_version();

	cudaMemcpy(host_version, dev_version, sizeof(Struct_Type), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(dev_version);

	return dev_version;


}

template <typename Struct_Type>
__host__ Struct_Type * move_to_device(Struct_Type * host_version, uint64_t num_copies){

	dev_version * get_device_version(num_copies);

	cudaMemcpy(dev_version, host_version, num_copies*sizeof(Struct_Type), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	cudaFree(host_version);

	return dev_version;


}

template <typename Struct_Type>
__host__ Struct_Type * move_to_host(Struct_Type * dev_version, uint64_t num_copies){

	host_version * get_host_version(num_copies);

	cudaMemcpy(host_version, dev_version, num_copies*sizeof(Struct_Type), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(dev_version);

	return dev_version;


}




// __device__ uint64_t reduce_less(cg::coalesced_threads active_threads, uint64_t val){

// 	int i = 1;



// }


}

}


#endif //GPU_BLOCK_