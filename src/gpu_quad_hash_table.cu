#ifndef GPU_QUAD_HASH_TABLE_CU
#define GPU_QUAD_HASH_TABLE_CU


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/hashutil.cuh"
#include "include/hash_metadata.cuh"

#include "include/gpu_quad_hash_table.cuh"
//#include <cuda_fp16.h>
#include <assert.h>



#if TAG_BITS == 16

	__device__ bool quad_hash_table::insert(uint16_t item)


#elif TAG_BITS == 32

	__device__ bool quad_hash_table::insert(uint item)

#elif TAG_BITS == 64

	__device__ bool quad_hash_table::insert(uint64_t item)

#endif

	{



		uint64_t primary_slot = MurmurHash64A((void *)&item, sizeof(item), seed) % num_slots;

		//this should implicitly wrap - don't want item to be key empty cause it could be overwritten
		if (item == KEY_EMPTY) item++;

		uint64_t slot = primary_slot;

		for (int j = 1 ; j < MAX_PROBE+1; j++){


			#if TAG_BITS == 16

			uint16_t old_key = atomicCAS((unsigned short int * )&slots[slot], KEY_EMPTY, (unsigned short int) item);

			#elif TAG_BITS == 32

			uint old_key = atomicCAS((unsigned int * )&slots[slot], KEY_EMPTY, (unsigned int) item);

			#elif TAG_BITS == 64

			uint64_t old_key = atomicCAS((unsigned long long int * )&slots[slot], KEY_EMPTY, (unsigned long long int) item);

			#endif

			if (old_key == KEY_EMPTY) return true;

			slot = (primary_slot + j*j) % num_slots;


		}


		return false;


	}


#if TAG_BITS == 16

	__device__ bool quad_hash_table::query(uint16_t item)


#elif TAG_BITS == 32

	__device__ bool quad_hash_table::query(uint item)

#elif TAG_BITS == 64

	__device__ bool quad_hash_table::query(uint64_t item)

#endif

	{

		uint64_t primary_slot = MurmurHash64A((void *)&item, sizeof(item), seed) % num_slots;

		//this should implicitly wrap - don't want item to be key empty cause it could be overwritten
		if (item == KEY_EMPTY) item++;

		uint64_t slot = primary_slot;

		for (int j = 1 ; j < MAX_PROBE+1; j++){


			if (slots[slot] == item) return true;

			slot = (primary_slot + j*j) % num_slots;

		}

		return false;

	}

 

	__global__ void init_table(quad_hash_table * table){

		uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

		if (tid >= table->num_slots) return;

		table->slots[tid] = KEY_EMPTY;

		return;
	}


	__host__ quad_hash_table * build_hash_table(uint64_t max_nitems){


		quad_hash_table * host_table;


		cudaMallocHost((void ** )&host_table, sizeof(quad_hash_table));

		host_table->num_slots = max_nitems;

		#if TAG_BITS == 16

		uint16_t * dev_slots;

		cudaMalloc((void **)&dev_slots, sizeof(uint16_t)*max_nitems);


		#elif TAG_BITS == 32

		uint * dev_slots;

		cudaMalloc((void **)&dev_slots, sizeof(uint)*max_nitems);

		#elif TAG_BITS == 64

		uint64_t * dev_slots;

		cudaMalloc((void **)&dev_slots, sizeof(uint64_t)*max_nitems);

		#endif

		host_table->slots = dev_slots;

		quad_hash_table * dev_table;

		cudaMalloc((void **)&dev_table, sizeof(quad_hash_table));

		cudaMemcpy(dev_table, host_table, sizeof(quad_hash_table), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		init_table<<<(max_nitems -1)/1024 + 1, 1024>>>(dev_table);

		cudaDeviceSynchronize();

		cudaFreeHost(host_table);

		return dev_table;

	}



#endif