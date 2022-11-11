/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/buddy_allocator.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace poggers::allocators;


//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   buddy_allocator<0,0> * first_allocator = buddy_allocator<0,0>::generate_on_device(2);


   buddy_allocator<1,0> * second_allocator = buddy_allocator<1,0>::generate_on_device(2);

   buddy_allocator<1,0> * third_allocator = buddy_allocator<1,0>::generate_on_device(2);


   cudaDeviceSynchronize();

   buddy_allocator<0,0>::free_on_device(first_allocator);

   buddy_allocator<1,0>::free_on_device(second_allocator);

   buddy_allocator<1,0>::free_on_device(third_allocator);

   cudaDeviceSynchronize();

   return 0;

}
