#ifndef MULTI_VQF_C
#define MULTI_VQF_C



#include <cuda.h>
#include <cuda_runtime_api.h>

#include "include/multi_vqf_host.cuh"
#include "include/atomic_vqf.cuh"

#include "include/metadata.cuh"
#include "include/hashutil.cuh"



//WORK IN PROGRESS
//DONT TRY TO COMPILE ME LOL


//this doesn't work because we can't actually read those values
//need 3 arrays for double buffering
// 1 ) host only
// 2 ) pinned host that has the addresses of the gpu stuff
// 3 ) gpu stuff


//TODO: double check that this index shit doesn't cause a segfault
//it shouldn't since we never actually peek at the memory
__host__ void multi_vqf::transfer_vqf_to_host(optimized_vqf * host, optimized_vqf * device, int stream){

		//buffers - start at byte 8
	cudaMemcpyAsync(host->buffers, device->buffers, sizeof(uint64_t)*host->num_blocks, cudaMemcpyDeviceToHost, streams[stream]);

	//buffer_sizes - start at byte 12
	cudaMemcpyAsync(host->buffer_sizes, device->buffer_sizes, sizeof(uint64_t)*host->num_blocks, cudaMemcpyDeviceToHost, streams[stream]);


	//team blocks  - start at byte 16

	cudaMemcpyAsync(host->blocks, device->blocks, sizeof(thread_team_block)*host->num_teams, cudaMemcpyDeviceToHost, streams[stream]);


}

__host__ void multi_vqf::transfer_vqf_from_host(optimized_vqf * host, optimized_vqf * device, int stream){


	//buffers - start at byte 8
	cudaMemcpyAsync(device->buffers, host->buffers, sizeof(uint64_t)*host->num_blocks, cudaMemcpyHostToDevice, streams[stream]);

	//buffer_sizes - start at byte 12
	cudaMemcpyAsync(device->buffer_sizes, host->buffer_sizes, sizeof(uint64_t)*host->num_blocks, cudaMemcpyHostToDevice, streams[stream]);


	//team blocks  - start at byte 16

	cudaMemcpyAsync(device->blocks, host->blocks, sizeof(thread_team_block)*host->num_teams, cudaMemcpyHostToDevice, streams[stream]);


}


__host__ void multi_vqf::transfer_to_host(int hostID, int activeID){


	//num_blocks and num_teams stay consitent across devices.
	//isolate and only transfer main data
	transfer_vqf_to_host(host_filters + hostID, device_filter_references + activeID, activeID);

}

__host__ void multi_vqf::transfer_to_device(int hostID, int activeID){

	transfer_vqf_from_host(host_filters + hostID, device_filter_references + activeID, activeID);


}

__host__ void multi_vqf::insert_into_filter(uint64_t * items, uint64_t nitems, int hostID, int activeID){


	transfer_to_device(hostID, activeID);

	device_filters[activeID].insert_async(items, nitems, host_filters[hostID].num_teams, host_filters[hostID].num_blocks, streams[activeID], misses);

	transfer_to_host(hostID, activeID);



}


__host__ multi_vqf::unload_active_blocks(){

	for (int i =0; i < active_filters; i++){


		cudaFree(device_filter_references[i].buffers);
		cudaFree(device_filter_references[i].buffer_sizes);
		cudaFree(device_filter_references[i].blocks);
	}
}

__host__ multi_vqf::load_active_blocks(){


	for (int i=0; i < active_filters; i++){


		uint64_t ** buffers;
		uint64_t * buffer_sizes;

		thread_team_block * blocks;


		cudaMalloc((void **)& buffers, sizeof(uint64_t *)*host_filters[0].num_blocks);
		cudaMalloc((void **)& buffer_sizes, sizeof(uint64_t)*host_filters[0].num_blocks);
		cudaMalloc((void **)& blocks, sizeof(thread_team_block)*host_filters[0].num_teams);


		device_filter_references[i].buffers = buffers;
		device_filter_references[i].buffer_sizes = buffer_sizes;
		device_filter_references[i].blocks = blocks;



	}

	//active filters are re-prepped
	cudaMemcpy(device_filters, device_filter_references, sizeof(optimized_vqf)*active_filters, cudaMemcpyHostToDevice);


}

__host__ multi_vqf::sort_batch(uint64_t * host_items, uint64_t nitems){


	//should be simple, async cudaFree on the blocks

	unload_active_blocks();

	const uint64_t block_size = 10000000000;


	uint64_t ** buffers;

	uint64_t ** buffer_sizes;


	uint64_t num_blocks = (nitems -1)/block_size +1;



	//buffers here must be offsets
	//since they reference a much larger host array
	//pointers are kind of pointless since they would all point to the same host object.
	cudaMallocHost((void **)& buffers, sizeof(uint64_t *)*num_blocks);
	cudaMallocHost((void **)& buffer_sizes, sizeof(uint64_t * )*num_blocks);

	for (int i =0; i < num_blocks; i++){

		uint64_t ** temp_buffers;
		uint64_t * temp_buffer_sizes;

		cudaMallocHost((void **)& temp_buffers, sizeof(uint64_t)*active_filters);
		cudaMallocHost((void **)& temp_buffer_sizes, sizeof(uint64_t)*active_filters);


		buffers[i] = temp_buffers;
		buffer_sizes[i] = temp_buffer_sizes;

	}


	uint64_t * cuda_buffer;

	uint64_t * buffers_slice;

	uint64_t * buffer_sizes_slice;

	cudaMalloc((void **)& cuda_buffer, sizeof(uint64_t)*block_size);


	cudaMalloc((void **)& buffers_slice, sizeof(uint64_t)*active_filters);
	cudaMalloc((void **)& buffer_sizes_slice, sizeof(uint64_t)*active_filters);



	//no double buffering here

	for (uint64_t i=0; i < num_blocks; i++){

		uint64_t items_in_batch = (i+block_size > nitems)? nitems-i: block_size;


		cudaMemcpyAsync(cuda_buffer, host_items+i, sizeof(uint64_t)*items_in_batch, cudaMemcpyHostToDevice, 0);



		//sort async uses hash -> hash 
		// it simplifies the scheme a lot and I think this is memory bound anyway - only transfer speed matters 
		sort_async(cuda_buffer, items_in_batch, buffers_slice, buffer_sizes_slice);


		//dump slices and hashes back into main memory

		cudaMemcpyAsync(buffers[i], buffers_slice, sizeof(uint64_t)*items_in_batch, cudaMemcpyDeviceToHost, 0)

		cudaMemcpyAsync(buffer_sizes[i], buffer_sizes_slice, sizeof(uint64_t)*items_in_batch, cudaMemcpyDeviceToHost, 0)

		


		cudaMemcpyAsync(host_items+i, cuda_buffer, sizeof(uint64_t)*items_in_batch, cudaMemcpyDeviceToHost, 0);


	}


	cudaFree()


	load_active_blocks();


}


__global__ void hash_all(uint64_t* vals, uint64_t* hashes, uint64_t nvals) {
	
	uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nvals){
		return;
	}

    uint64_t key = vals[idx];


    //does this need to be clipped?
    //I think NOT
    // this doesn't map to specific slots yet so not necessary
    key = MurmurHash64A(((void *)&key), sizeof(key), 42);

    hashes[idx] = key;

	return;

}


__host__ multi_vqf::sort_async(uint64_t * item_block, uint64_t nitems, uint64_t * buffers, uint64_t * buffer_sizes){


	hash_all<<<(nitems-1)/1024 + 1, 1024>>>(item_block, item_block, nitems);


	//replace with something better
	thrust::sort(thrust::device, item_block, item_block+nvals); 

	attach_buffers


}

__host__ multi_vqf * build_vqf(int num_filters, int bits_per_filter){


	multi_vqf * host_vqf;

	cudaMallocHost((void **)& host_vqf, sizeof(multi_vqf));


	optimized_vqf * host_sections;

	optimized_vqf * dev_sections;

	optimized_vqf * dev_sections_host_ref;


	cudaStream_t * streams;

	//active
	cudaMallocHost((void **) & streams, active_filters * sizeof(cudaStream_t));

	for(int i=0; i< active_filters; i++){

		cudaStreamCreate(&(streams[i]));
	}


	host_vqf->streams = streams;

	cudaMallocHost((void ** ) & host_sections, num_filters*sizeof(optimized_vqf));
	cudaMallocHost((void ** ) & dev_sections_host_ref, active_filters*sizeof(optimized_vqf));


	for (int i=0; i < num_filters; i++){

		host_sections[i] = * prep_host_vqf(1ULL << bits_per_filter);


	}


	//todo: modify this to be dev_num_filters - need to select optimal number of dev filters
	cudaMalloc((void **) & dev_sections, active_filters*sizeof(optimized_vqf));


	for (int i =0; i< active_filters; i++){

		optimized_vqf * dev_i = build_vqf(1ULL << bits_per_filter);

		cudaMemcpy(dev_sections + i, dev_i, sizeof(optimized_vqf), cudaMemcpyDeviceToDevice);

	}


	cudaMemcpy(dev_sections_host_ref, dev_sections, active_filters*sizeof(optimized_vqf), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();


	host_vqf->host_filters = host_sections;
	host_vqf->device_filters = dev_sections;
	host_vqf->device_filter_references = dev_sections_host_ref;

	for (int i =0; i < num_filters; i++){

		host_vqf->transfer_to_host(i,0);
	}

	cudaDeviceSynchronize();

	//setup prepped hosts here

	uint64_t * misses;

	cudaMallocManaged((void ** ) &misses, sizeof(uint64_t));

	misses[0] = 0;

	host_vqf->misses = misses;



	return host_vqf;
}


#endif