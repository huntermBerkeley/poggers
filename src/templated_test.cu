/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>



#include "include/key_val_pair.cuh"
#include "include/templated_block.cuh"
#include "include/metadata.cuh"
#include "include/templated_vqf.cuh"
#include "include/templated_sorting_funcs.cuh"

#include <openssl/rand.h>



__host__ uint64_t * generate_data(uint64_t nitems){


	//malloc space

	uint64_t * vals = (uint64_t *) malloc(nitems * sizeof(uint64_t));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(uint64_t));



		to_fill += togen;

		printf("Generated %llu/%llu\n", to_fill, nitems);

	}

	return vals;
}


__host__ uint64_t * load_main_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/main_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(uint64_t));

	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(uint64_t), pFile);

	if (result != nitems*sizeof(uint64_t)) abort();



	// //current supported format is no spacing one endl for the file terminator.
	// if (myfile.is_open()){


	// 	getline(myfile, line);

	// 	strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

	// 	myfile.close();
		

	// } else {

	// 	abort();
	// }


	return (uint64_t *) vals;


}

__host__ uint64_t * load_alt_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/fp_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(uint64_t));


	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(uint64_t), pFile);

	if (result != nitems*sizeof(uint64_t)) abort();



	return (uint64_t *) vals;


}

__host__ void test_key_val_pairs(){

	printf("test_storage: %d, expect 2\n", sizeof(key_val_pair<test_struct, test_struct>));
	printf("test / empty: %d, expect 1\n", sizeof(key_val_pair<test_struct, empty>));
	printf("uint8_t Empty_storage: %d, expect 1\n", sizeof(key_val_pair<uint8_t, empty>));
	printf("uint16_t Empty_storage: %d, expect 2\n", sizeof(key_val_pair<uint16_t, empty>));
	printf("uint32_t Empty_storage: %d, expect 4\n", sizeof(key_val_pair<uint32_t, empty>));
	printf("uint64_t Empty_storage: %d, expect 8\n\n", sizeof(key_val_pair<uint64_t, empty>));

	printf("uint8_t and nothing: %d, expect 1\n", sizeof(key_val_pair<uint8_t>));
	printf("uint16_t and nothing: %d, expect 2\n", sizeof(key_val_pair<uint16_t>));
	printf("uint32_t and nothing: %d, expect 4\n", sizeof(key_val_pair<uint32_t>));
	printf("uint64_t and nothing: %d, expect 8\n\n", sizeof(key_val_pair<uint64_t>));

	printf("uint8_t and uint32_t: %d, expect 5\n", sizeof(key_val_pair<uint8_t, uint32_t, wrapper>));
	printf("uint16_t and uint32_t: %d, expect 6\n", sizeof(key_val_pair<uint16_t, uint32_t, wrapper>));
	printf("uint32_t and uint32_t: %d, expect 8\n", sizeof(key_val_pair<uint32_t, uint32_t, wrapper>));
	printf("uint64_t and uint32_t: %d, expect 12\n\n", sizeof(key_val_pair<uint64_t, uint32_t, wrapper>));


	//	key_pair %d, space_available %d, block %d\n", sizeof(uint16_t), sizeof(key_val_pair<uint16_t>), BYTES_AVAILABLE, sizeof(templated_block<key_val_pair<uint16_t>>));



	//key_val_pair<uint8_t> test(4);

	key_val_pair<uint8_t, empty, empty_wrapper> test_pair();

	key_val_pair<uint8_t, empty, empty_wrapper> test_pair_2(4);
	//printf("Value in test %d\n", test.get_key());
	
	printf("Storage pair val: %d\n", test_pair_2.get_key());

	key_val_pair<uint8_t, uint32_t, wrapper> full (8,12);

	printf("Storage pair val: %d, %d\n", full.get_key(), full.get_val());
}


template <typename key_val_pair>
__global__ void insert_test_kernel(templated_block<key_val_pair> * blocks, key_val_pair * buffer_keys, int buffer_count, key_val_pair * primary_items, int primary_nitems, key_val_pair * secondary_items, int secondary_nitems, const uint64_t dividing_line, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	int warpID = tid % 32;

	blocks[0].dump_all_buffers_sorted(buffer_keys, buffer_count, primary_items, primary_nitems, secondary_items, secondary_nitems, 0, warpID, dividing_line);

	assert(assert_sorted<key_val_pair>(buffer_keys, 3));


	//check that keys were inserted
	blocks[0].sorted_bulk_query(9, warpID, buffer_keys, hits, 3);

	assert(hits[0] == true);
	assert(hits[1] == true);
	assert(hits[2] == true);

	blocks[0].sorted_bulk_query(9, warpID, primary_items, hits, 3);

	assert(hits[0] == true);
	assert(hits[1] == true);
	assert(hits[2] == true);

	blocks[0].sorted_bulk_query(9, warpID, secondary_items, hits, 3);

	assert(hits[0] == true);
	assert(hits[1] == true);
	assert(hits[2] == true);



	assert(blocks[0].binary_search_query(buffer_keys[2], 9));


	return;
}



template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
__global__ void sorting_funcs_kernel(key_val_pair<Key, Val, Wrapper> * items, int nitems){

	int warpID = threadIdx.x % 32;

	sorting_network<Key, Val, Wrapper>(items, nitems, warpID);


	bool val = assert_sorted<key_val_pair<Key, Val, Wrapper>>(items, nitems);

	assert(val);

	//assert(assert_sorted<key_val_pair<Key,Val, Wrapper>>(items, nitems)   );
 
}

__host__ void test_sorting_funcs(){


	using key_type = uint8_t;


	key_val_pair<key_type> * keys;

	cudaMallocManaged((void **)&keys, 32*sizeof(key_type));

	int cap = 1;

	for (int i=0; i < cap; i++){

		keys[i].set_key( (uint8_t) ((125*i+1) % 199));
	}


	sorting_funcs_kernel<key_type><<<1,32>>>(keys, cap);

	cudaDeviceSynchronize();

	printf("%d done.\n", cap);

	fflush(stdout);


	cap = 4;

	for (int i=0; i < cap; i++){

		keys[i].set_key( (uint8_t) ((125*i+1) % 199));
	}


	sorting_funcs_kernel<key_type><<<1,32>>>(keys, cap);

	cudaDeviceSynchronize();

	printf("%d done.\n", cap);

	fflush(stdout);

	cap = 8;

	for (int i=0; i < cap; i++){

		keys[i].set_key( (uint8_t) ((125*i+1) % 199));
	}


	sorting_funcs_kernel<key_type><<<1,32>>>(keys, cap);

	cudaDeviceSynchronize();

	printf("%d done.\n", cap);

	fflush(stdout);


	cap = 16;

	for (int i=0; i < cap; i++){

		keys[i].set_key( (uint8_t) ((125*i+1) % 199));
	}


	sorting_funcs_kernel<key_type><<<1,32>>>(keys, cap);

	cudaDeviceSynchronize();

	printf("%d done.\n", cap);

	fflush(stdout);


	cap = 32;

	for (int i=0; i < cap; i++){

		keys[i].set_key( (uint8_t) ((125*i+1) % 199));
	}


	sorting_funcs_kernel<key_type><<<1,32>>>(keys, cap);

	cudaDeviceSynchronize();

	printf("%d done.\n", cap);

	fflush(stdout);


}


int main(int argc, char** argv) {
	

	//uint64_t nbits = atoi(argv[1]);

	//uint64_t num_batches = atoi(argv[2]);

	//double batch_percent = 1.0 / num_batches;

	//test_key_val_pairs();

	test_sorting_funcs();

	using val_type = uint8_t;

	using key = key_val_pair<val_type, val_type, wrapper>;



	const uint64_t dividing_line = (1ULL << (3*sizeof(val_type)));

	printf("dividing_line: %llu\n", dividing_line);



	key * buffer_keys;


	cudaMallocManaged((void **)& buffer_keys, sizeof(key)*10);

	key * primary_keys;


	cudaMallocManaged((void **)& primary_keys, sizeof(key)*10);


	key * secondary_keys;


	cudaMallocManaged((void **)& secondary_keys, sizeof(key)*10);


	templated_block<key> * blocks;


	cudaMallocManaged((void **)&blocks, sizeof(templated_block<key>));


	bool * hits;

	cudaMallocManaged((void **)&hits, 10*sizeof(bool));


	cudaDeviceSynchronize();


	//copy and paste to init keys
	

	//printf("Storage pair val: %d, %d\n", buffer_keys[0].get_key(), buffer_keys[0].get_val());

	//buffer_keys[0].set_key(1);

	printf("Bytes per key: %d\n", sizeof(key));

	buffer_keys[0].pack_into_pair((uint16_t) 1, (uint16_t) 3);
	buffer_keys[1].pack_into_pair(2, 2);
	buffer_keys[2].pack_into_pair(3, 1);


	primary_keys[0].pack_into_pair(1, 3);
	primary_keys[1].pack_into_pair(2, 4);
	primary_keys[2].pack_into_pair(3, 5);

	secondary_keys[0].pack_into_pair(1, 5);
	secondary_keys[1].pack_into_pair(2, 6);
	secondary_keys[2].pack_into_pair(3, 7);





	printf("buf_0 %d, buf_1 %d\n", buffer_keys[0].get_val(), buffer_keys[1].get_val());
	fflush(stdout);

	assert(buffer_keys[0].get_key() < buffer_keys[1].get_key());
	assert(buffer_keys[0] < buffer_keys[1]);



	printf("Size of block: %d\n", blocks[0].max_size());


	//Val_Key_storage_pair<test_struct, test_struct> test;

	//Val_Key_storage_pair<uint8_t, num_wrapper<uint8_t>> other_test;

	cudaDeviceSynchronize();

	insert_test_kernel<key><<<1,32>>>(blocks, buffer_keys, 3, primary_keys, 3, secondary_keys, 3, dividing_line, hits);

	cudaDeviceSynchronize();

	//End of key_val_stuff

	uint64_t * test_ints;

	key_val_pair<uint8_t> * test_keys;

	cudaMallocManaged((void **)& test_ints, 10*sizeof(uint64_t));


	cudaMallocManaged((void **)& test_keys, 10*sizeof(key_val_pair<uint8_t>));


	uint64_t * misses;

	cudaMallocManaged((void **)&misses, sizeof(uint64_t));


	// bool * hits;

	// cudaMallocManaged((void **)&hits, 10*sizeof(bool))



	cudaDeviceSynchronize();

	misses[0] = 0;

	test_ints[0] = 1;
	test_ints[1] = 2;
	test_ints[2] = 3;
	test_ints[3] = 0;
	test_ints[4] = 0;

	test_keys[0].set_key(5);
	test_keys[1].set_key(6);
	test_keys[2].set_key(7);
	test_keys[3].set_key(8);
	test_keys[4].set_key(9);


	cudaDeviceSynchronize();


	auto vqf = build_vqf<uint8_t>(4000);

	vqf->attach_lossy_buffers(test_ints, test_keys, 5, vqf->get_num_blocks());

	vqf->bulk_insert(misses, vqf->get_num_teams());


	cudaDeviceSynchronize();

	printf("Misses: %llu\n", misses[0]);

	misses[0] = 0;

	test_ints[0] = 1;
	test_ints[1] = 2;
	test_ints[2] = 3;
	test_ints[3] = 0;
	test_ints[4] = 0;

	test_keys[0].set_key(5);
	test_keys[1].set_key(6);
	test_keys[2].set_key(7);
	test_keys[3].set_key(8);
	test_keys[4].set_key(9);


	for (int i = 0; i < 10; i++){

		hits[i] = false;
	}

	cudaDeviceSynchronize();


	//this name sucks but 
	vqf->attach_lossy_buffers(test_ints, test_keys, 5, vqf->get_num_blocks());

	vqf->bulk_query(hits, vqf->get_num_teams());


	cudaDeviceSynchronize();


	for (int i=0; i < 10; i++){

		printf("%d ", hits[i]);

	}

	printf("\n");

	for (int i=0; i < 3; i++){

		assert(hits[i]);

	}



	cudaDeviceSynchronize();

	free_vqf(vqf);

	cudaDeviceSynchronize();


	test_sorting_funcs();

	cudaDeviceSynchronize();

	return 0;

}
