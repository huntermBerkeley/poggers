
#ifndef TEMPLATED_BLOCK_CU
#define TEMPLATED_BLOCK_CU


#include <cuda.h>
#include <cuda_runtime_api.h>
//#include "include/templated_block.cuh"
#include "include/warp_utils.cuh"
#include "include/metadata.cuh"
#include "include/sorting_helper.cuh"

//extra stuff
#include <cuda.h>
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

//VQF Block
// Functions Required:

// Lock();
// get_fill();
// Unlock_other();
// Insert();
// Unlock();

//I'm putting the bit manipulation here atm



//set the original 1 bits of the block
//this is done on a per thread level

template <typename Tag_type>
void templated_block<Tag_type>::test(){

	printf("I have space for %d items!\n", sizeof(tags));

	return;
}


#endif //atomic_block_CU