#ifndef GLOBAL_LOAD_VQF_CU
#define GLOBAL_LOAD_VQF_CU


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/hashutil.cuh"
#include "include/hash_metadata.cuh"

#include "include/warp_storage_block.cuh"
#include "include/global_load_vqf.cuh"
//#include <cuda_fp16.h>
#include <assert.h>

#if TAG_BITS == 8

#define mask 0xff

#elif TAG_BITS == 16

#define mask 0xffff

#elif TAG_BITS == 32

#define mask 0xffffffff

#elif TAG_BITS == 64

//don't judge me
#define mask 0xffffffffffffffffULL

#endif



__global__ void bulk_insert_kernel(optimized_vqf * vqf, uint64_t * items, uint64_t nitems, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t itemID = tid / 32;

	if (itemID >= nitems) return;

	int warpID = tid % 32;

	bool inserted = vqf->insert_item(items[itemID], warpID);

	if (!inserted){



		uint64_t hash = vqf->get_hash(items[itemID]) % vqf->num_blocks;

		
		int counter = atomicAdd(vqf->counters+hash,0);

		if (vqf->counters[hash] < 32){

			vqf->insert_item(items[itemID], warpID);
		}

		assert(atomicAdd(vqf->counters+hash,0) == 32);

		if (warpID == 0){

			atomicAdd((unsigned long long int *) misses, (unsigned long long int) 1);



		}
		

	}

}


__host__ void optimized_vqf::bulk_insert(uint64_t * items, uint64_t nitems, uint64_t * misses){


	bulk_insert_kernel<<<32*nitems -1 /1024 + 1, 1024>>>(this, items, nitems, misses);


}



__global__ void bulk_query_kernel(optimized_vqf * vqf, uint64_t * items, uint64_t nitems, bool * hits){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t itemID = tid / 32;

	if (itemID >= nitems) return;

	uint64_t warpID = tid % 32;

	hits[itemID] = vqf->query_item(items[itemID], warpID);

}




__host__ void optimized_vqf::bulk_query(uint64_t * items, uint64_t nitems, bool * hits){

	bulk_query_kernel<<<32*nitems-1/1024+1,1024>>>(this, items, nitems, hits);

}

__device__ uint64_t optimized_vqf::get_hash(uint64_t item){

	item = MurmurHash64A(((void *)&item), sizeof(item), seed);
	return item;

}


__device__ bool optimized_vqf::insert_item(uint64_t item, int warpID){


	uint64_t hash = get_hash(item) % num_blocks;

	bool inserted = blocks[hash].insert(item & mask, warpID);

	if (inserted){

		if (warpID == 0)
		atomicAdd((unsigned int *) counters+hash, (unsigned int) 1);

		return true;
	} else {

		return false;

	}

	


}

__device__ bool optimized_vqf::query_item(uint64_t item, int warpID){

	uint64_t hash = get_hash(item) % num_blocks;

	return blocks[hash].query(item & mask, warpID);
}


//allocate a vqf with at least nslots
__host__ optimized_vqf * build_vqf(uint64_t nslots){

	uint64_t num_blocks = (nslots -1) / 32 + 1;

	optimized_vqf * host_vqf;

	cudaMallocHost((void **)& host_vqf, sizeof(optimized_vqf));

	storage_block * blocks;

	cudaMalloc((void **)& blocks, num_blocks*sizeof(storage_block));

	cudaMemset(blocks, 0, num_blocks*sizeof(storage_block));

	uint * counters;


	cudaMalloc((void **)&counters, num_blocks*sizeof(uint));

	cudaMemset(counters, 0, num_blocks*sizeof(uint));


	host_vqf->num_blocks = num_blocks;

	host_vqf->blocks = blocks;

	host_vqf->counters = counters;

	//wow what a seed
	host_vqf->seed = 43;

	optimized_vqf * dev_vqf;

	cudaMalloc((void **)&dev_vqf, sizeof(optimized_vqf));

	cudaMemcpy(dev_vqf, host_vqf, sizeof(optimized_vqf), cudaMemcpyHostToDevice);

	cudaFreeHost(host_vqf);

	return dev_vqf;

}

__host__ void free_vqf(optimized_vqf * vqf){


	optimized_vqf * host_vqf;

	cudaMallocHost((void **)& host_vqf, sizeof(optimized_vqf));

	cudaMemcpy(host_vqf, vqf, sizeof(optimized_vqf), cudaMemcpyDeviceToHost);

	cudaFree(vqf);

	cudaFree(host_vqf->blocks);

	cudaFree(host_vqf->counters);

	cudaFreeHost(host_vqf);



}
 


#endif