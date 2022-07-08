#ifndef FREE_LIST
#define FREE_LIST


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/allocators/superblock.cuh>
#include <poggers/allocators/base_heap_ptr>



// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };


//a pointer list managing a set section o fdevice memory

namespace poggers {


namespace allocators { 


struct free_list {

	public:

		uint64_t total_size;

		heap_wrapper internal_heap;

		__device__ memory_ptr * get_global_footer(){

			char * converted_this = (char *) this;

			return (memory_ptr *) (converted_this + sizeof(free_list) + total_size - sizeof(memory_ptr));


		}

		__device__ memory_ptr * get_global_head(){

			char * converted_this = (char * ) this;

			return (memory_ptr *) (converted_this + sizeof(free_list));
		}

		//find the previous node in the list
		//this is used for freeing and tracks 
		__device__ heap_wrapper find_prev_node(){


			memory_ptr * next_node;

			if (internal_heap.footer->next() >= internal_heap.footer){

				//point to the end of the list


				next_node = get_global_footer();


			} else {

				next_node = internal_heap.

			}



		}






		__host__ __device__ free(){};


		__host__ global_heap * generate_on_device(uint64_t num_bytes){

			void * byte_space;

			cudaMalloc((void **)&byte_space, num_bytes);

			def_free_list = base_heap_ptr(byte_space, num_bytes);

			global_heap host_heap;

			host_heap.free_list = def_free_list;

			global_heap * dev_heap;

			cudaMemcpy(dev_heap, &host_heap, sizeof(global_heap), cudaMemcpyHostToDevice);

		}


		__host__ free_on_device(global_heap * dev_heap){

			global_heap host_heap;

			cudaMemcpy(&host_heap, dev_heap, sizeof(global_heap), cudaMemcpyDeviceToHost);

			base_heap_ptr::free_on_device(host_heap.free_list);


		}

		

};



}

}


#endif //GPU_BLOCK_