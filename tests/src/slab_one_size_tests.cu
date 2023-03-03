/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/slab_one_size.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#include <cooperative_groups.h>


#include <poggers/allocators/one_size_allocator.cuh>

namespace cg = cooperative_groups;

using namespace poggers::allocators;


__global__ void malloc_tests(one_size_slab_allocator<15> * allocator, uint64_t max_mallocs){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= max_mallocs) return;

   void * allocation = allocator->malloc();

   return;

}


__host__ void boot_slab_one_size(){


   one_size_slab_allocator<15> * test_alloc = one_size_slab_allocator<15>::generate_on_device(64000000, 16);

   cudaDeviceSynchronize();

   malloc_tests<<<1, 256>>>(test_alloc, 10);

   cudaDeviceSynchronize();


   one_size_slab_allocator<15>::free_on_device(test_alloc);

   cudaDeviceSynchronize();

}



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {


   for (int i =0; i< 20; i++){
      boot_slab_one_size();
   }
   

 
   cudaDeviceReset();
   return 0;

}
