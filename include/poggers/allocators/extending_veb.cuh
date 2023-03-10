#ifndef EXT_VEB_TREE
#define EXT_VEB_TREE
//A CUDA implementation of the Extending Van Emde Boas tree, made by Hunter McCoy (hunter@cs.utah.edu)
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


//The Extending Van Emde Boas Tree, or EVEB, is a data structure that supports efficient data grouping and allocation/deallocation based on use.
//given a target size and a memory chunk size, the tree dynamically pulls/pushes chunks to the free list based on usage.
//the metadata supports up to the maximum size passed in, and persists so that the true structure does not mutate over the runtime.


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



//thank you interwebs https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << line  << ":" << std::endl << file  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

#define EXT_VEB_RESTART_CUTOFF 30

#define EXT_VEB_GLOBAL_LOAD 1
#define EXT_VEB_MAX_ATTEMPTS 15

namespace poggers {

namespace allocators {


template<uint64_t bytes_per_chunk, uint64_t alloc_size>
struct extending_veb_allocator {


	static_assert(bytes_per_chunk % alloc_size == 0);


	__host__ uint64_t get_max_veb_chunks(){

		size_t mem_total;
		size_t mem_free;
		cudaMemGetInfo  (&mem_free, &mem_total);

		return mem_total;
	}




}



}

}


#endif //End of VEB guard