#ifndef WARP_STORAGE_CU
#define WARP_STORAGE_CU


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/hashutil.cuh"
#include "include/hash_metadata.cuh"

#include "include/warp_storage_block.cuh"
//#include <cuda_fp16.h>
#include <assert.h>



#if TAG_BITS == 16

	__device__ bool storage_block::insert(uint16_t item, int warpID)


#elif TAG_BITS == 32

	__device__ bool storage_block::insert(uint item, int warpID)

#elif TAG_BITS == 64

	__device__ bool storage_block::insert(uint64_t item, int warpID)

#endif

	{

		//extra l2 load stuff - use me if problems
		// .L2::128B - im broken but some specification includes something like this

		uint64_t * my_address = slots+warpID;

		#if TAG_BITS == 16

		uint16_t my_slot;
		
		asm("ld.global.u16 %0, [%1];" : "=h"(my_slot) : "l"(my_address));

		#elif TAG_BITS == 32

		uint my_slot;

		asm("ld.global.u32 %0, [%1];" : "=r"(my_slot) : "l"(my_address));

		#elif TAG_BITS == 64

		uint64_t my_slot;

		asm("ld.global.u64 %0, [%1];" : "=l"(my_slot) : "l"(my_address));
		//my_slot = slots[warpID];

		#endif

		//uint64_t my_slot;
		
		__syncwarp();

		bool ballot = (my_slot == KEY_EMPTY);
		
		int result = __ffs(__ballot_sync(0xffffffff, ballot)) -1;

		//if (result == warpID)
		//broadcast


		while (result != -1){


			if (result == warpID){

			#if TAG_BITS == 16

			uint16_t old_key = atomicCAS((unsigned short int * )&slots[warpID], KEY_EMPTY, (unsigned short int) item);

			#elif TAG_BITS == 32

			uint old_key = atomicCAS((unsigned int * )&slots[warpID], KEY_EMPTY, (unsigned int) item);

			#elif TAG_BITS == 64

			uint64_t old_key = atomicCAS((unsigned long long int * )&slots[warpID], KEY_EMPTY, (unsigned long long int) item);

			#endif

			if (old_key != KEY_EMPTY) ballot = false;



		} 

		bool succeeded = __shfl_sync(0xffffffff, ballot, result);

		if (succeeded) return true;

		//on failure retry
		result = __ffs(__ballot_sync(0xffffffff, ballot)) -1;






		}


		if (result == -1) return false;


	}


#if TAG_BITS == 16

	__device__ bool storage_block::query(uint16_t item, int warpID)


#elif TAG_BITS == 32

	__device__ bool storage_block::query(uint item, int warpID)

#elif TAG_BITS == 64

	__device__ bool storage_block::query(uint64_t item, int warpID)

#endif

	{


		uint64_t * my_address = slots+warpID;


		#if TAG_BITS == 16

		uint16_t my_slot;
		
		asm("ld.global.u16 %0, [%1];" : "=h"(my_slot) : "l"(my_address));

		#elif TAG_BITS == 32

		uint my_slot;

		asm("ld.global.u32 %0, [%1];" : "=r"(my_slot) : "l"(my_address));

		#elif TAG_BITS == 64

		uint64_t my_slot;

		//my_slot = slots[warpID];


		asm("ld.global.u64 %0, [%1];" : "=l"(my_slot) : "l"(my_address));

		#endif
		
		__syncwarp();

		bool ballot = (my_slot == item);
		
		int result = __ffs(__ballot_sync(0xffffffff, ballot)) -1;

		//if (result == warpID)
		//broadcast

		if (result != -1) return true;

		return false;

	}

 


#endif