#ifndef CUDA_QUEUE 
#define CUDA_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/metadata.cuh"
#include "include/key_val_pair.cuh"
#include "include/templated_block.cuh"
#include "include/hashutil.cuh"
#include "include/templated_sorting_funcs.cuh"
#include <stdio.h>
#include <assert.h>
#include <cooperative_groups.h>

//thrust stuff
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>


//counters are now external to allow them to permanently reside in the l1 cache.
//this should improve performance and allow for different loading schemes
//that are less reliant on the initial load.

//these are templated on just one thing
//key_value_pairs

// template <typename Tag_type>
// __device__ bool assert_sorted(Tag_type * tags, int nitems){


// 	if (nitems < 1) return true;

// 	Tag_type smallest = tags[0];

// 	for (int i=1; i< nitems; i++){

// 		if (tags[i] < smallest) return false;

// 		smallest = tags[i];
// 	}

// 	return true;

// }





// template <typename Filter, typename Key_type>
// __global__ void test_kernel(T * my_vqf){


// }

template <typename T>
struct submission_block {

	uint64_t submissionID;

	int submission_type;

	uint64_t work_done;

	T ** buffers;

	int * buffer_sizes;



};


template<typename T>
struct cuda_queue {


	//tag bits change based on the #of bytes allocated per block
	submission_block<T> * blocks;
	
	uint64_t max_tasks;

	

	uint64_t current_enqueued_num;

	//uint64_t current_task_num;

	//cudaStream_t peristent_stream, setup_stream;


	//these bad boys are exact!
	//__host__ bulk_insert(key_type * keys);

	//check if an items is ready to be processed
	__device__ bool current_object_ready(int id_to_query){

		assert(id_to_query > 0);


		int slot_to_check = id_to_query % max_tasks;

		//int submittedID = blocks[slot_to_check].submissionID

		// uint64_t * my_address = &blocks[slot_to_check].submissionID;

		// uint64_t current_id;
		
		// asm("ld.global.u64 %0, [%1];" : "=l"(current_id) : "l"(my_address));

		uint64_t current_id = atomicCAS((unsigned long long int *)&blocks[slot_to_check].submissionID, 0,0);

		if (current_id == id_to_query) return true;

		//if less we are at least one loop away
		if (current_id < id_to_query) return false;

		if (current_id > id_to_query){

			printf("Primary ID passed, make sure the queue is large enough / Hunter add a check here\n");
			printf("current_id %llu, id_to_query %d, block says %llu\n", current_id, id_to_query, blocks[slot_to_check].submissionID);
			assert(current_id <= id_to_query);
		}

		return false; 

	}

	//precondition - assumes block has been loaded successfully
	//and is waiting load
	__device__ submission_block<T> * load_current_block(int id_to_query){


		int slot_to_check = id_to_query % max_tasks;

		#if DEBUG_ASSERTS



		assert(blocks[slot_to_check].submissionID == id_to_query);

		#endif

		return &blocks[slot_to_check];

	}


	__device__ void submit_task(submission_block<T> * block){


		uint64_t my_write_slot = atomicAdd((unsigned long long int *) &current_enqueued_num, 1ULL) % max_tasks;

		assert(blocks[my_write_slot].work_done);

		blocks[my_write_slot].buffers = block->buffers;
		blocks[my_write_slot].buffer_sizes = block->buffer_sizes;
		blocks[my_write_slot].submission_type = block->submission_type;
		blocks[my_write_slot].work_done = false;

		//force write
		__threadfence();

		atomicExch((unsigned long long int *) &blocks[my_write_slot].submissionID, (unsigned long long int) block->submissionID);

		return;

	}

	__device__ void sychronize(){


		cooperative_groups::grid_group g = cooperative_groups::this_grid();
		g.sync();

	}


	__device__ bool task_done(int id_to_query){


		int slot_to_check = id_to_query % max_tasks;

		//int submittedID = blocks[slot_to_check].submissionID

		//uint64_t * my_address = &blocks[slot_to_check].work_done;

		uint64_t current_id;
		
		//asm("ld.global.u64 %0, [%1];" : "=l"(current_id) : "l"(my_address));

		current_id = atomicCAS((unsigned long long int *)&blocks[slot_to_check].work_done, 0,0);

		return current_id;

	}


};




template<typename T>
__host__ void free_queue(cuda_queue<T> * queue){


	cuda_queue<T> * host_queue;


	cudaMallocHost((void **)& host_queue, sizeof(cuda_queue<T>));

	cudaMemcpy(host_queue, queue, sizeof(cuda_queue<T>), cudaMemcpyDeviceToHost);

	cudaFree(queue);

	cudaFree(host_queue->blocks);



	cudaFreeHost(host_queue);




}

template <typename T>
__global__ void prep_queue(cuda_queue<T> * queue){

	uint64_t tid = threadIdx.x+blockDim.x*blockIdx.x;

	if (tid > 0) return;

	for (int i=0; i < queue->max_tasks; i++){

		queue->blocks[i].work_done = false;
		queue->blocks[i].submissionID = 0;
	}
}


template<typename T>
__host__ submission_block<T> * get_queue_head(cuda_queue<T> * queue){

	cuda_queue<T> * host_queue;

	cudaMallocHost((void **)&host_queue, sizeof(cuda_queue<T>));

	cudaMemcpy(host_queue, queue, sizeof(cuda_queue<T>), cudaMemcpyDeviceToHost);

	submission_block<T> * head = &(host_queue->blocks[0]);

	cudaFreeHost(host_queue);

	return head;

}


template<typename T>
__host__ cuda_queue<T>  * build_queue(uint64_t nitems){




	cuda_queue<T> * host_queue;

	cudaMallocHost((void **)&host_queue, sizeof(cuda_queue<T>));

	
	host_queue->max_tasks = nitems;

	//host_queue->current_task_num = 1;
	host_queue->current_enqueued_num = 1;


	submission_block<T> * blocks;



	cudaMalloc((void **)& blocks, nitems*sizeof(submission_block<T>));


	host_queue->blocks = blocks;

	//this should 
	//buffers

	//create streams
	

	cuda_queue<T> * queue;


	cudaMalloc((void **)& queue, sizeof(cuda_queue<T>));

	cudaMemcpy(queue, host_queue, sizeof(cuda_queue<T>), cudaMemcpyHostToDevice);

	cudaFreeHost(host_queue);


	prep_queue<<<1,1>>>(queue);



	return queue;

}


template<typename T>
__global__ void persistent_kernel_test(cuda_queue<T> * queue){


	uint64_t tid = threadIdx.x+blockDim.x*blockIdx.x;

	int current_item = 1;

	assert(tid == 0);

	while (true){



		//perform a long read to scout out

		if (queue->current_object_ready(current_item)){


			//printf("Executing %llu/%d!\n", tid, current_item);

			submission_block<T> * current_block = queue->load_current_block(current_item);

			if (current_block->submission_type==2){
				//continue;
				//printf("task %d should be silent\n", current_item);
			}

			else if (current_block->submission_type == 1){
				printf("Task %d is a print task!\n", current_item);

			} else if (current_block->submission_type == 0){
				printf("Task %d is ending\n", current_item);

				return;
			}

			if (tid == 0){

				atomicExch((unsigned long long int *) &current_block->work_done, 1ULL);
				//current_block->work_done = true;

			}

			__threadfence();

			//queue->sychronize();

			__syncthreads();

			current_item+=1;

		}




	}


}


#endif //GPU_BLOCK_