#ifndef ONE_SIZE_ALLOCATOR
#define ONE_SIZE_ALLOCATOR
//A CUDA implementation of the Van Emde Boas tree, made by Hunter McCoy (hunter@cs.utah.edu)
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


//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <poggers/allocators/veb.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


namespace poggers {

namespace allocators {


struct one_size_allocator{

	uint64_t size_per_alloc;
	void * allocation;
	veb_tree * internal_tree;
	


	static __host__ one_size_allocator * generate_on_device(uint64_t universe, uint64_t ext_size_per_alloc, uint64_t seed){


		one_size_allocator * host_alloc;


		cudaMallocHost((void **)&host_alloc, sizeof(one_size_allocator));

		cudaDeviceSynchronize();

		host_alloc->internal_tree = veb_tree::generate_on_device(universe, seed);
		void * dev_allocation;

		cudaMalloc((void **)&dev_allocation, ext_size_per_alloc*universe);

		cudaMemset(dev_allocation, 0, ext_size_per_alloc*universe);

		cudaDeviceSynchronize();

		host_alloc->allocation = dev_allocation;

		host_alloc->size_per_alloc = ext_size_per_alloc;

		one_size_allocator * dev_allocator;

		cudaMalloc((void ** )&dev_allocator, sizeof(one_size_allocator));

		cudaMemcpy(dev_allocator, host_alloc, sizeof(one_size_allocator), cudaMemcpyHostToDevice);


		cudaFreeHost(host_alloc);

		return dev_allocator;


	}

	static __host__ void free_on_device(one_size_allocator * dev_version){

		one_size_allocator * host_alloc;


		cudaMallocHost((void **)&host_alloc, sizeof(one_size_allocator));

		cudaDeviceSynchronize();

		cudaMemcpy(host_alloc, dev_version, sizeof(one_size_allocator), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(host_alloc->allocation);

		cudaFree(dev_version);

		veb_tree::free_on_device(host_alloc->internal_tree);

		cudaFreeHost(host_alloc);


	}

	__device__ void * malloc(){

		uint64_t offset = internal_tree->malloc();

		//return (void *) offset;

		if (offset == veb_tree::fail()) return nullptr;

		uint64_t address = (uint64_t) allocation + offset*size_per_alloc;

		return (void * ) address;

	}


	__device__ void free(void * ext_allocation){

		//internal_tree->insert((uint64_t) ext_allocation);

		uint64_t offset = ((uint64_t) ext_allocation - (uint64_t) allocation)/size_per_alloc;

		internal_tree->insert(offset);

	}


	__device__ uint64_t get_offset(){

		return internal_tree->malloc();

	}


	__device__ void free_offset(uint64_t offset){

		internal_tree->insert(offset);
	}

	__device__ void * get_mem_from_offset(uint64_t offset){

		uint64_t address = (uint64_t) allocation + offset*size_per_alloc;

		return (void *) address;

	}

	__device__ uint64_t get_offset_from_address(void * address){

		uint64_t offset = ((uint64_t) ext_allocation - (uint64_t) allocation)/size_per_alloc;

		return offset;

	}


};



}

}


#endif //End of VEB guard