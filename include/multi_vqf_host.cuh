#ifndef MULTI_VQF_H
#define MULTI_VQF_H



#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/atomic_vqf.cuh"

#include "include/metadata.cuh"


//doesn't need to be explicitly packed
typedef struct __attribute__ ((__packed__)) multi_vqf {


	int num_filters;

	int active_filters;

	optimized_vqf * host_filters;

	optimized_vqf * device_filters;

	optimized_vqf * device_filter_references;
	
	cudaStream_t * streams;

	uint64_t * misses;

	__host__ void transfer_vqf_to_host(optimized_vqf * host, optimized_vqf * device, int stream);

	__host__ void transfer_vqf_from_host(optimized_vqf * host, optimized_vqf * device, int stream);


	__host__ void transfer_to_host(int hostID, int activeID);

	__host__ void transfer_to_device(int hostID, int activeID);

	__host__ void insert_into_filter(uint64_t * items, uint64_t nitems, int hostID, int activeID);


	__host__ unload_active_blocks();

	__host__ load_active_blocks();


} multi_vqf;

 


__host__ multi_vqf * build_vqf(int num_filters, int bits_per_filter);


#endif