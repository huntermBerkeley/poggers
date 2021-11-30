

#ifndef WARP_UTILS
#define WARP_UTILS




#include <cuda.h>

#include <cuda_runtime_api.h>


namespace warp_utils {

__device__ unsigned int uint_pext(int laneId, unsigned int reduce, unsigned int deposit);

__device__ uint64_t uint64_pext(int laneId, uint64_t reduce, uint64_t deposit);

__device__ uint64_t uint64_pdep(int laneId, uint64_t src, uint64_t mask);

__device__ void printUint(unsigned int val);

//print a unsigned int as binary
__device__ void printUint64_t(uint64_t val);


__device__ int select(int warpID, uint64_t val, int bit);


__device__ void warp_memmove(int warpID, void* dst, const void* src, size_t n);

__device__ void block_8_memmove_insert(int warpID, uint16_t * tags, uint16_t tag, int index);

__device__ void block_8_memmove_remove(int warpID, uint16_t * tags, int index);


}




#endif