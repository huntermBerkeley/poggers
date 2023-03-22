#ifndef ALLOC_TABLE
#define ALLOC_TABLE
//A CUDA implementation of the alloc table, made by Hunter McCoy (hunter@cs.utah.edu)
//Copyright (C) 2023 by Hunter McCoy

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
//and associated documentation files (the "Software"), to deal in the Software without restriction, 
//including without l> imitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
//LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//The alloc table is an array of uint64_t, uint64_t pairs that store



//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/sub_veb.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {


//alloc table associates chunks of memory with trees

//using uint16_t as there shouldn't be that many trees.

//register atomically inserst tree num, or registers memory from chunk_tree.


template<uint64_t bytes_per_chunk>
struct alloc_table {


	using my_type = alloc_table<bytes_per_chunk>;

	uint16_t * chunk_ids;

	static __host__ my_type * generate_on_device(){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));


		uint64_t num_chunks = poggers::utils::get_max_chunks<bytes_per_chunk>();

		printf("Booting memory table with %llu chunks\n", num_chunks);

		uint16_t * ext_chunks;

		cudaMalloc((void **)&ext_chunks, sizeof(uint16_t)*num_chunks);


		cudaMemset(ext_chunks, ~0U, sizeof(uint16_t)*num_chunks);

		host_version->chunk_ids = ext_chunks;

		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_version);


		return dev_version;


	}

	static __host__ my_type * generate_on_device(uint64_t max_bytes){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));


		uint64_t num_chunks = poggers::utils::get_max_chunks<bytes_per_chunk>(max_bytes);

		printf("Booting memory table with %llu chunks\n", num_chunks);

		uint16_t * ext_chunks;

		cudaMalloc((void **)&ext_chunks, sizeof(uint16_t)*num_chunks);


		cudaMemset(ext_chunks, ~0U, sizeof(uint16_t)*num_chunks);

		host_version->chunk_ids = ext_chunks;

		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_version);


		return dev_version;


	}

	static __host__ void free_on_device(my_type * dev_version){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);


		cudaDeviceSynchronize();

		cudaFree(host_version->chunk_ids);

		cudaFree(dev_version);

		cudaFreeHost(host_version);


	}

	//register a tree component
	__device__ void register_tree(uint64_t block, uint16_t id){

		chunk_ids[block] = id;

	}

	//register a block from the table.
	__device__ void register_size(uint64_t block, uint16_t size){

		size+=16;

		chunk_ids[block] = size;

	}

	__device__ void get_tree_size(uint64_t block, uint16_t&id, uint16_t&size){
		return;
	}



};



}

}


#endif //End of VEB guard