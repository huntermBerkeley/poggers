#ifndef MULTI_ALLOCATOR
#define MULTI_ALLOCATOR
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
#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif

namespace poggers {

namespace allocators {


//alloc table associates chunks of memory with trees

//using uint16_t as there shouldn't be that many trees.

//register atomically inserst tree num, or registers memory from chunk_tree.


template<uint64_t bytes_per_chunk, uint64_t smallest, uint64_t biggest>
struct multi_allocator {


	using my_type = multi_allocator<bytes_per_chunk, smallest, biggest>;
	using sub_tree_type = extending_veb_allocator_nosize<bytes_per_chunk>;

	one_size_allocator * bit_tree;
	one_size_allocator * chunk_tree;

	alloc_table<bytes_per_chunk> * table;

	sub_tree_type ** sub_trees;

	static __host__ my_type * generate_on_device(uint64_t max_bytes, uint64_t seed){


		my_type * host_version;

		cudaMalloc((void **)&host_version, sizeof(my_type));


		//plug in to get max chunks
		uint64_t max_chunks = poggers::utils::get_max_chunks<bytes_per_chunk>(max_bytes);

		host_version->chunk_tree = one_size_allocator::generate_on_device(max_chunks, bytes_per_chunk, seed);


		//estimate the max_bits
		uint64_t num_bits = max_bytes/smallest;

		uint64_t num_bytes = 0;

		do {

			num_bytes += ((num_bits -1)/64+1)*8;

			num_bits = num_bits/64;
		} while (num_bits > 64);



		//need to verify, but this should be sufficient for all sizes.
		host_version->bit_tree = poggers::utils::generate_on_device(max_chunks, num_bytes, seed);


		uint64_t num_trees = get_first_bit_bigger(biggest) - get_first_bit_bigger(smallest);

		sub_tree_type ** ext_sub_trees = poggers::utils::get_device_version<sub_tree_type *>(num_trees);

		for (int i = 0; i < num_trees; i++){

			ext_sub_trees[i] = sub_tree_type::generate_on_device(get_p2_from_index(get_first_bit_bigger(smallest)+i), seed+i, max_bytes);

		}

		host_version->sub_trees = poggers::utils::move_to_device<sub_tree_type *>(ext_sub_trees, num_trees);

		host_version->table = alloc_table<bytes_per_chunk>::generate_on_device(max_bytes);

	}


	//return the index of the largest bit set
	__host__ int get_first_bit_bigger(uint64_t counter){

	//	if (__builtin_popcountll(counter) == 1){

			//0th bit would give 63

			//63rd bit would give 0
			return 63 - __builtin_clzll(counter) + (__builtin_popcountll(counter) == 1);

	}

	__device__ int get_first_bit_bigger(uint64_t counter){

		return 63 - __clzll(counter) + (__popcll(counter) == 1);
	}


	__host__ __device__ uint64_t get_p2_from_index(int index){


		return (1ULL) << index;


	}

	static __host__ void free_on_device(my_type * dev_version){



	}




};



}

}


#endif //End of VEB guard