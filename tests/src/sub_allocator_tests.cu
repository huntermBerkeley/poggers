/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */


#define DEBUG_ASSERTS 0

#define DEBUG_PRINTS 1



#include <poggers/allocators/sub_allocator.cuh>
#include <poggers/allocators/free_list.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>

#define stack_bytes 32768




using allocator = poggers::allocators::sub_allocator_wrapper<stack_bytes, 4>::sub_allocator_type;
//using allocator = poggers::allocators::sub_allocator<stack_bytes, 7>;

using global_ptr = poggers::allocators::header;


__global__ void test_allocator_one_thread(global_ptr * heap){


   //this test size is illegal I think
   //but....
   //we have a dynamic memory manager
   //so lets just request from that lol
   const uint64_t test_size = 32000;


   uint ** addresses = (uint **) heap->malloc_aligned(test_size*sizeof(uint *), 16, 0);

   allocator * new_allocator = allocator::init(heap);

   for (int i = 0; i < test_size; i++){

      addresses[i] = (uint *) new_allocator->malloc(4, heap);

   }


   for (int i = 0; i < test_size; i++){

      if (addresses[i] != nullptr){
          new_allocator->stack_free(addresses[i]);
      }
     
   }

   heap->free(addresses);

   allocator::free_allocator(heap, new_allocator);


}

__global__ void test_allocator_variations(global_ptr * heap){


   using allocator_4 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 4>::sub_allocator_type;

   using allocator_8 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 8>::sub_allocator_type;

   using allocator_16 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 16>::sub_allocator_type;

   using allocator_32 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 32>::sub_allocator_type;

   using allocator_64 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 64>::sub_allocator_type;

   using allocator_128 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 128>::sub_allocator_type;

   using allocator_256 = poggers::allocators::sub_allocator_wrapper<stack_bytes, 256>::sub_allocator_type;

   allocator_4 * new_allocator_4 = allocator_4::init(heap);
   allocator_4::free_allocator(heap, new_allocator_4);

   allocator_8 * new_allocator_8 = allocator_8::init(heap);
   allocator_8::free_allocator(heap, new_allocator_8);

   allocator_16 * new_allocator_16 = allocator_16::init(heap);
   allocator_16::free_allocator(heap, new_allocator_16);

   allocator_32 * new_allocator_32 = allocator_32::init(heap);
   allocator_32::free_allocator(heap, new_allocator_32);

   allocator_64 * new_allocator_64 = allocator_64::init(heap);
   allocator_64::free_allocator(heap, new_allocator_64);

   allocator_128 * new_allocator_128 = allocator_128::init(heap);
   allocator_128::free_allocator(heap, new_allocator_128);

   allocator_256 * new_allocator_256 = allocator_256::init(heap);
   allocator_256::free_allocator(heap, new_allocator_256);




}

int main(int argc, char** argv) {


   //allocate 
   const uint64_t bytes_in_use = 800000;

   global_ptr * heap = global_ptr::init_heap(bytes_in_use);

   cudaDeviceSynchronize();

   test_allocator_one_thread<<<1,1>>>(heap);

   cudaDeviceSynchronize();

   test_allocator_variations<<<1,1>>>(heap);

   cudaDeviceSynchronize();

   global_ptr::free_heap(heap);

   cudaDeviceSynchronize();


}
