

#ifndef mega_vqf_C
#define mega_vqf_C



#include "include/mega_vqf.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/megablock.cuh"

#include <iostream>

#include <fstream>
#include <assert.h>


#define MAX_FILL 28

__device__ void mega_vqf::lock_block(int warpID, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 0,1) != 0);	
	// }
	// __syncwarp();

	//TODO: turn me back on
	blocks[lock].lock(warpID);
}

__device__ void mega_vqf::unlock_block(int warpID, uint64_t lock){


	// if (warpID == 0){

	// 	while(atomicCAS(locks + lock, 1,0) != 1);	

	// }

	// __syncwarp();

	blocks[lock].unlock(warpID);
}

__device__ void mega_vqf::lock_blocks(int warpID, uint64_t lock1, uint64_t lock2){


	if (lock1 < lock2){

		lock_block(warpID, lock1);
		lock_block(warpID, lock2);
		//while(atomicCAS(locks + lock2, 0,1) == 1);

	} else {


		lock_block(warpID, lock2);
		lock_block(warpID, lock1);
		
	}

	


}

__device__ void mega_vqf::unlock_blocks(int warpID, uint64_t lock1, uint64_t lock2){


	if (lock1 > lock2){

		unlock_block(warpID, lock1);
		unlock_block(warpID, lock2);
		
	} else {

		unlock_block(warpID, lock2);
		unlock_block(warpID, lock1);
	}
	

}

__device__ bool mega_vqf::insert(int warpID, uint64_t hash){

   uint64_t block_index = (hash >> TAG_BITS) % num_blocks;



   //this will generate a mask and get the tag bits
   uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;

   // assert(block_index < num_blocks);


   //external locks
   //blocks[block_index].extra_lock(block_index);
 	
 	lock_block(warpID, block_index);
 	int side_fill = blocks[block_index].get_fill();

 	//side_fill < 20 ||

 	if (block_index == alt_block_index){


 		if (side_fill < MAX_FILL){

 			blocks[block_index].insert(warpID, hash);


 			unlock_block(warpID, block_index);

 			return true;

 		} else {

 			unlock_block(warpID, block_index);
 			return false;
 		}

 		


 		

 	}


 	unlock_block(warpID, block_index);


 	lock_blocks(warpID, block_index, alt_block_index);

   int fill_main = blocks[block_index].get_fill();

   int fill_alt = blocks[alt_block_index].get_fill();


   bool toReturn = false;

   if (fill_main < fill_alt){


   	unlock_block(warpID, alt_block_index);



   	//if (fill_main < SLOTS_PER_BLOCK-1){
   	if (fill_main < MAX_FILL){
   		blocks[block_index].insert(warpID, hash);

   		toReturn = true;

   	}

   	unlock_block(warpID, block_index);


   } else {

   	unlock_block(warpID, block_index);

   	if (fill_alt < MAX_FILL){

	   	blocks[alt_block_index].insert(warpID, hash);

	   	toReturn = true;

	   }

	   unlock_block(warpID, alt_block_index);

   }



 	//unlock_blocks(block_index, alt_block_index);


   return toReturn;





}


__device__ bool mega_vqf::query(int warpID, uint64_t hash){

	uint64_t block_index = (hash >> TAG_BITS) % num_blocks;

   //this will generate a mask and get the tag bits
   uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;

   if (block_index == alt_block_index){

   	lock_block(warpID, block_index);



   	bool found = blocks[block_index].query(warpID, hash);

   	unlock_block(warpID, block_index);

   	return found;


   }

   lock_blocks(warpID, block_index, alt_block_index);


   bool found = blocks[block_index].query(warpID, hash) || blocks[alt_block_index].query(warpID, hash);


   unlock_blocks(warpID, block_index, alt_block_index);

   return found;

}


__device__ bool mega_vqf::remove(int warpID, uint64_t hash){


	uint64_t block_index = (hash >> TAG_BITS) % num_blocks;

   //this will generate a mask and get the tag bits
   uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
   uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;


   lock_block(warpID, block_index);

   bool found = blocks[block_index].remove(warpID, hash);

   unlock_block(warpID, block_index);

   //copy could be deleted from this instance

   if (found){
   	return true;
   }

   lock_block(warpID, alt_block_index);

   found = blocks[alt_block_index].remove(warpID, hash);


   unlock_block(warpID, alt_block_index);

   return found;

}


// __device__ bool mega_vqf::insert(uint64_t hash){

//    uint64_t block_index = (hash >> TAG_BITS) % num_blocks;



//    //this will generate a mask and get the tag bits
//    uint64_t tag = hash & ((1ULL << TAG_BITS) -1);
//    uint64_t alt_block_index = (((hash ^ (tag * 0x5bd1e995)) % (num_blocks*SLOTS_PER_BLOCK)) >> TAG_BITS) % num_blocks;

//    assert(block_index < num_blocks);


//    //external locks
//    //blocks[block_index].extra_lock(block_index);
   
//    while(atomicCAS(locks + block_index, 0, 1) == 1);



//    int fill_main = blocks[block_index].get_fill();


//    if (fill_main >= SLOTS_PER_BLOCK-1){

//    	while(atomicCAS(locks + block_index, 0, 1) == 0);
//    	//blocks[block_index].unlock();

//    	return false;
//    }

//    if (fill_main < .75 * SLOTS_PER_BLOCK || block_index == alt_block_index){
//    	blocks[block_index].insert(tag);

   	

//    	int new_fill = blocks[block_index].get_fill();
//    	if (new_fill != fill_main+1){
//    		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
//    		assert(blocks[block_index].get_fill() == fill_main+1);
//    	}


//    	while(atomicCAS(locks + block_index, 1, 0) == 0);
//    	//blocks[block_index].unlock();
//    	return true;
//    }


//    while(atomicCAS(locks + block_index, 1, 0) == 0);

//    lock_blocks(block_index, alt_block_index);


//    //need to grab other block

//    //blocks[alt_block_index].extra_lock(alt_block_index);
//    while(atomicCAS(locks + alt_block_index, 0, 1) == 1);

//    int fill_alt = blocks[alt_block_index].get_fill();

//    //any larger and we can't protect metadata
//    if (fill_alt >=  SLOTS_PER_BLOCK-1){
// //   	blocks[block_index.unlock()]

//    	unlock_blocks(block_index, alt_block_index);
//    	//blocks[alt_block_index].unlock();
//    	//blocks[block_index].unlock();
//    	return false;
//    }


//    //unlock main
//    if (fill_main > fill_alt ){

//    	while(atomicCAS(locks + block_index, 1, 0) == 0);
//    	//blocks[block_index].unlock();

//    	blocks[alt_block_index].insert(tag);
//    	assert(blocks[alt_block_index].get_fill() == fill_alt+1);

//    	int new_fill = blocks[alt_block_index].get_fill();
//    	if (new_fill != fill_alt+1){
//    		printf("Broken Fill: Block %llu, old %d new %d\n", alt_block_index, fill_alt, new_fill);
//    		assert(blocks[alt_block_index].get_fill() == fill_alt+1);
//    	}

//    	while(atomicCAS(locks + alt_block_index, 1, 0) == 0);
//    	//blocks[alt_block_index].unlock();


//    } else {

//    	while(atomicCAS(locks + alt_block_index, 1, 0) == 0);
//    	//blocks[alt_block_index].unlock();
//    	blocks[block_index].insert(tag);

//    	int new_fill = blocks[block_index].get_fill();
//    	if (new_fill != fill_main+1){
//    		printf("Broken Fill: Block %llu, old %d new %d\n", block_index, fill_main, new_fill);
//    		assert(blocks[block_index].get_fill() == fill_main+1);
//    	}

//    	while(atomicCAS(locks + block_index, 1, 0) == 0);
//    	//blocks[block_index].unlock();

//    }


  
//    return true;



//}


__global__ void mega_vqf_block_setup(mega_vqf * mega_vqf){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= mega_vqf->num_blocks) return;

	mega_vqf->blocks[tid].setup();

}

__host__ mega_vqf * build_mega_vqf(uint64_t nitems){


	//this seems weird but whatever
	uint64_t num_blocks = (nitems -1)/SLOTS_PER_BLOCK + 1;


	printf("Bytes used: %llu for %llu blocks.\n", num_blocks*sizeof(megablock),  num_blocks);


	mega_vqf * host_mega_vqf;

	mega_vqf * dev_mega_vqf;

	megablock * blocks;

	cudaMallocHost((void ** )& host_mega_vqf, sizeof(mega_vqf));

	cudaMalloc((void ** )& dev_mega_vqf, sizeof(mega_vqf));	

	//init host
	host_mega_vqf->num_blocks = num_blocks;

	//allocate blocks
	cudaMalloc((void **)&blocks, num_blocks*sizeof(megablock));

	cudaMemset(blocks, 0, num_blocks*sizeof(megablock));

	host_mega_vqf->blocks = blocks;


	//external locks
	int * locks;
	cudaMalloc((void ** )&locks, num_blocks*sizeof(int));
	cudaMemset(locks, 0, num_blocks*sizeof(int));


	host_mega_vqf->locks = locks;



	cudaMemcpy(dev_mega_vqf, host_mega_vqf, sizeof(mega_vqf), cudaMemcpyHostToDevice);

	cudaFreeHost(host_mega_vqf);

	mega_vqf_block_setup<<<(num_blocks - 1)/64 + 1, 64>>>(dev_mega_vqf);
	cudaDeviceSynchronize();

	return dev_mega_vqf;


}

#endif

