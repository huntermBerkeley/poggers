/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>

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


// __global__ void test_kernel(veb_tree * tree, uint64_t num_removes, int num_iterations){


//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid >= num_removes)return;


//       //printf("Tid %lu\n", tid);


//    for (int i=0; i< num_iterations; i++){


//       if (!tree->remove(tid)){
//          printf("BUG\n");
//       }

//       tree->insert(tid);

//    }

template <uint64_t mem_segment_size, uint64_t num_bits>
__host__ void boot_ext_tree(){

   using tree_type = extending_veb_allocator_nosize<mem_segment_size>;

   tree_type * tree_to_boot = tree_type::generate_on_device(num_bits, 1342);

   cudaDeviceSynchronize();

   //tree_type::free_on_device(tree_to_boot);

   cudaDeviceSynchronize();

}


template <uint64_t mem_segment_size>
__host__ void boot_alloc_table(){


   using table_type = alloc_table<mem_segment_size>;

   table_type * table = table_type::generate_on_device();

   cudaDeviceSynchronize();

   table_type::free_on_device(table);

}
// }

// __global__ void view_kernel(veb_tree * tree){

//    uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

//    if (tid != 0)return;



// }



//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   boot_ext_tree<8ULL*1024*1024, 16ULL>();
 
   boot_ext_tree<8ULL*1024*1024, 4096ULL>();


   boot_alloc_table<8ULL*1024*1024>();

   cudaDeviceReset();
   return 0;

}
