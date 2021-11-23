#ifndef _VQF_BLOCK_ 
#define _VQF_BLOCK_


#include <cuda.h>
#include <cuda_runtime_api.h>






#define TAG_BITS 16


#if TAG_BITS == 8

//need to reconcile # blocks to nums
#define SLOTS_PER_BLOCK 48
#define VIRTUAL_BUCKETS 80

#elif TAG_BITS == 16

#define SLOTS_PER_BLOCK 28
#define VIRTUAL_BUCKETS 36 

#endif



#define LOCK_MASK (1ULL << 63)

#define UNLOCK_MASK ~(1ULL << 63)

typedef struct __attribute__ ((__packed__)) vqf_block {


	//metadata and tags change based on the size of 
	//tag bits
	#if TAG_BITS == 8

		volatile uint64_t md[2];
		uint8_t tags[48];

	#elif TAG_BITS == 16

		
		volatile uint64_t md[1];
		uint16_t tags[28];

	#endif

	__device__ void setup();


	__device__ void lock(int warpID);



	__device__ void unlock(int warpID);
	__device__ int max_capacity();

	__device__ uint64_t shift_upper_bits(uint64_t bits, int cutoff);
	__device__ uint64_t get_upper_bit(uint64_t bits);
	__device__ void md_0_and_shift_right(int index);
	__device__ int get_fill();


	__device__ void insert(int warpID, uint64_t item);

	__device__ bool query(int warpID, uint64_t item);

	__device__ bool remove(int warpID, uint64_t item);
	//remove related functions
	__device__ uint64_t get_lower_bit(uint64_t bits);


	__device__ uint64_t shift_lower_bits(uint64_t bits, int cutoff);

	__device__ void down_shift(int index);


	__device__ void printBlock();

	__device__ void printMetadata();


} vqf_block;

// #if TAG_BITS == 8
// 	// We are using 8-bit tags.
// 	// One block consists of 48 8-bit slots covering 80 buckets, and 80+48 = 128
// 	// bits of metadata.
// 	typedef struct __attribute__ ((__packed__)) vqf_block {
// 		uint64_t md[2];
// 		uint8_t tags[48];

// 		void test();
// 	} 
// #elif TAG_BITS == 12
// 	// We are using 12-bit tags.
// 	// One block consists of 32 12-bit slots covering 96 buckets, and 96+32 = 128
// 	// bits of metadata.
//         // NOTE: not supported yet.
// 	typedef struct __attribute__ ((__packed__)) vqf_block {
// 		uint64_t md[2];
// 		uint8_t tags[32]; // 32 12-bit tags
  
// 		void test();
// 	} vqf_block;
// #elif TAG_BITS == 16 
// 	// We are using 16-bit tags.
// 	// One block consists of 28 16-bit slots covering 36 buckets, and 36+28 = 64
// 	// bits of metadata.
// 	typedef struct __attribute__ ((__packed__)) vqf_block {
// 		uint64_t md;
// 		uint16_t tags[28];

// 		void test();
// 	} vqf_block;
// #endif


//DEFINE FUNCS



#endif //_VQF_BLOCK_