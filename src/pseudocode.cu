/* pseudocode for the bulked power-of-two-choice hashing

- The main idea is to use the sorted lists to generate an expected fill each bucket
	* Each item can then compare between it's expected fills and pick the lower bucket
	* This will act as a heuristic that will always overshoot the expected fill
	* Further testing is needed to determine if this is a viable approach and/or affects variance

*/




//Coming into the algorithm, we need space for items and the pointers to be sorted
//lets assume that all of these are set up
uint64_t nitems;
uint64_t nbuffers;
uint64_t * items;
short int * first_slots;
short int * second_slots;  


//these is setup as x[i] = i; - these are necessary to relect back to the main array
uint64_t * indices;


//this is setup as x[i] = len(x) + i;
uint64_t * alt_indices;

//these arrays are necessary but can be initialized to 0;
uint64_t * hashes;
uint64_t * alt_hashes;
uint64_t * returnedIndices;
uint64_t * returnedAltIndices;

vqf * main_vqf;

//the buffers - on reflection I think these can be rolled into the algorithm to save space
uint64_t * buffers;
uint64_t * buffer_sizes;

uint64_t * alt_buffers;
uint64_t * alt_buffer_sizes;



//and some functions that aren't directly related

//these generate1⁄
uint64_t get_bucket_from_hash(uint64_t hash);
uint64_t get_alt_bucket_from_hash(uint64_t hash);

//assign the buffers and buffer sizes
// after this buffers[i] is a list of buffer_sizes[i] items
void assign_buffers(uint64_t * buffers, uint64_t * buffer_sizes, uint64_t * sorted_list);




void bulk_insert(){


	for (i =0; i < nitems; i++) in parallel {

		hashes[i] = get_bucket_from_hash(items[i]);
		alt_hashes[i] = get_alt_bucket_from_hash(items[i]);

	}

	//combine the two hashes into one large array
	uint64_t * combined_array;

	combined_array[0:nitems] = hashes;

	combined_array[nitems:2*nitems] = alt_hashes;


	uint64_t * combined_indices;

	combined_indices[0:nitems] = indices;

	combined_indices[nitems:2*nitems] = alt_indices;


	thrust::sort_by_key(combined_array, combined_indices);

	//thrust::sort_by_key(alt_hashes, alt_indices);

	//at this point each of the hashes are in order

	assign_buffers(buffers, buffer_sizes, hashes);
	assign_buffers(alt_buffers, alt_buffer_sizes, alt_hashes);


	for (j=0; j < nbuffers; j++) in parallel{



		for (i =0; i < buffer_sizes[j]; i++){

			returnedIndices[indices[buffers[j] + i]] = i + main_vqf.blocks[j].get_fill();

		}

	}


	//problem - these two can point to the same location :( - best solutuion is to interleave

		

		

}