#ifndef GLOBAL_HEAP
#define GLOBAL_HEAP


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <variant>

#include <poggers/allocators/superblock.cuh>
#include <poggers/allocators/base_heap_ptr>


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 1
#endif


#define LOCK_MASK = (1ULL << 63);
#define ALLOCED_MASK = (1ULL << 62);
#define ENDPOINT_MASK = (1ULL << 63);

#define CUTOFF_SIZE 1024

// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

namespace poggers {


namespace allocators { 


struct mixed_counter {
	uint first;
	uint second;
	//uint64_t  offset : 59; 

	bool assert_valid(){
		return (first == second);
	}

};

union large_unioned_bitfield {

	mixed_counter as_bitfield;
	uint64_t as_uint;

};


struct header_lock_dist {


	//bit-flags

	//0: locked
	//1: allocated
	//2: endpoint


	uint16_t lock : 3;
	uint64_t size : 61;

}

union header_bitfield {

	header_lock_dist as_fields;

	uint64_t as_uint;

	__device__ bool lock(){

		return !(atomicOR(&this, LOCK_MASK) & LOCK_MASK);

	}

	__device__ void unlock(){

		atomicAND(&lock_and_size, ~LOCK_MASK);

		return;
	}


};
	


struct atomicLock{

	header_bitfield * header_to_lock;

	inline atomicLock(header_bitfield * ext_bitfield){

		header_to_lock = ext_bitfield;

		while (!header_to_lock->lock());

	}

	inline ~atomicLock(){

		 header_to_lock->unlock();

	}

}

struct header{

	header_bitfield lock_and_size;
	uint64_t empty;
	header * next;
	header * prev;


	__device__ bool lock(){


		return lock_and_size.lock();

	}

	__device__ void stall_lock(){

		while (!header_to_lock=>lock());

	}

	__device__ void unlock(){


		return lock_and_size.unlock();		

	}

	__device__ header_bitfield atomic_get(){

		return atomicOR(&lock_and_size, 0);
	}


	__device__ bool alloc(){

		return atomicOR(&lock_and_size, ALLOCED_MASK);
	}

	__device__ bool dealloc(){

		return atomicAND(&lock_and_size, ~ALLOCED_MASK);
	}


	__device__ bool replace(header_bitfield old, header_bitfield new){


		return (atomicCAS(&lock_and_size, old.as_uint, new_status.as_uint) == old.as_uint);

	}

	__device__ bool check_alloc(){


		return atomic_get() & ALLOCED_MASK;

	}

	__device__ bool check_endpoint(){

		return atomic_get() & ENDPOINT_MASK;

	}

	__device__ void set_size(uint64_t new_size){

		lock_and_size.as_bitfield.size = new_size;
		return 
	}

	__device__ uint64_t get_size(){
		return lock_and_size.as_bitfield.size;
	}

	__device__ void set_next(header * next_node){
		next = next_node;
	}

	__device__ void set_prev(header * prev_node){
		prev = prev_node;
	}


	__device__ header * get_next(){
		return next;
	}

	__device__ header * get_prev(){
		return prev;
	}

	__device__ header * init_node(void * allocation, uint64_t num_bytes){

		header * node = (header *) allocation;

		node->lock_and_size.as_bitfield.lock = 0;
		node->lock_and_size.as_bitfield.size = num_bytes;


		return node;

	}


	__device__ void init_heap(void * allocation, uint64_t num_bytes){

		header * heap_start = init_node(allocation, 32);
		header * main_node = init_node(allocation+32, num_bytes-64);
		header * heap_end = init_node(allocation+num_bytes-32, 32);

		heap_start->lock_and_size.as_uint ^= ENDPOINT_MASK;
		heap_end->lock_and_size.as_uint ^= ENDPOINT_MASK;

		heap_start->set_next(main_node);
		main_node->set_next(heap_end);
		heap_end->set_next(heap_start);

		heap_start->set_prev(heap_end);
		main_node->set_prev(heap_start);
		heap_end->set_prev(main_node);

		return heap_start;


	}


	__device__ void split_non_locking(uint64_t num_bytes){


		//atomicLock myLock(&lock_and_size);

		header * next_node = get_next();

		//atomicLock nextLock(&next_node->lock_and_size);


		lock_and_size.as_bitfield.size -= num_bytes;

		void * next_address = ((void *) this) + lock_and_size.as_bitfield.size;

		header * new_node = init_node(next_address, num_bytes);

		this->set_next(new_node);

		new_node->set_next(next_node);


		new_node->set_prev(this);

		next_node->set_prev(new_node);

		return;


	}

	//dequeue a node
	__device__ static void *  allocate_next(header * prev, header * main, header * next){


		prev->set_next(next);
		next->set_prev(prev);


		main->alloc();

		void * address = (void *) main;

		prev->unlock();
		next->unlock();
		return address+32;


	}



	__device__ void split(uint64_t num_bytes){


		atomicLock myLock(&lock_and_size);

		header * next_node = get_next();

		atomicLock nextLock(&next_node->lock_and_size);


		lock_and_size.as_bitfield.size -= num_bytes;

		void * next_address = ((void *) this) + lock_and_size.as_bitfield.size;

		header * new_node = init_node(next_address, num_bytes);

		this->set_next(new_node);

		new_node->set_next(next_node);


		new_node->set_prev(this);

		next_node->set_prev(new_node);

		return;


	}


	__device__ void remove_from_list(){

		header * prev = get_prev();

		header * next = get_next();

		prev->set_next(next);

		next->set_prev(prev);

		return;

	}


	//get the next node in the list
	//this is acheived by pointing to the next node
	__device__ inline header * get_next_disallocated(){

		void * this_void = (void *) this;

		this_void = this_void + num_bytes;

		return (header *) this_void;


	}



	//merge two blocks together
	//assumes already locked
	__device__ void merge(){


		#if DEBUG_ASSERTS

		void * this_next = get_next_disallocated();

		void * next = (void *) get_next();

		assert(this_next == next);

		assert (get_next()->get_prev() == this);

		#endif

		header * next = get_next();

		next->remove_from_list();

		lock_and_size.as_bitfield.size += next->lock_and_size.as_bitfield.as_bitfield.size;


	}




	//find the first node participating in the free list
	//find next free assume unlocked
	//returns the head locked
	__device__ header * find_next_free(){


		//atomicLock head_lock(&lock_and_size);

		//assume head already locked


		bool alloced = check_alloc();
		header * head = this;
		header * new_head;

		//head->lock();

		bool alloced = false;

		while (!alloced){

			new_head = head->get_next();

			new_head->lock();


			//maintain lock on original node - don't want to free
			if (head != this){
				head->unlock();
			}
			

			head = new_head;

			alloced = head->check_alloc();

		}

		return head;

	}


	__device__ header * cycle_to_next_node(){

		header * next = get_next();

		next->stall_lock();

		unlock();

		return next;

	}

	__device__ void * find_first_fit(uint64_t num_bytes){

		

		header * prev = this;

		prev->stall_lock();

		header * main = get_next();

		main->stall_lock();

		header * next = main->get_next();

		next->stall_lock();

		header * potential_next = next->get_next();

		//do a full loop
		while (potential_next != this){

			if (main.get_size() >= num_bytes){


				//detach!

				uint64_t leftover = main.get_size() - num_bytes;

				if (leftover > CUTOFF_SIZE){

					//split

					ideal_node = main->split_non_locking(num_bytes);

					ideal_node->remove_from_list();

					prev->unlock();
					main->unlock();
					next->unlock();



				} else {

					main.remove_from_list();


					prev->unlock();
					main->unlock();
					next->unlock();


				}

			}

			potential_next->stall_lock();
			prev->unlock();

			prev = main;
			main = next;
			next = potential_next;
			potential_next = next->get_next();


		}


		//full failure release locks
		prev->unlock();
		main->unlock();
		next->unlock();

		return nullptr;



	}

	//regular malloc call
	__device__ void * malloc(uint64_t num_bytes){

		//need to find a block with at least this header
		uint64_t bytes_needed = num_bytes+32;

		return find_first_fit(bytes_needed);


	}


	//given head -> next locked
	// grab next->next and merge
	__device__ header * merge_next_node(header * next){

		if (get_next_disallocated() == next && !next->check_endpoint()){


			next_next = next->get_next();

			//we get this eventually
			next_next->stall_lock();

			next->merge();

			return next_next();



		}

		return next;


	}

	//absorbs all the nodes to the right that can be absorbed
	// maintains the locks
	__device__ header * merge_all_left(header * next){


		header * new_next = merge_next_node(next);

		while (new_next != next ){

			next = new_next;

			new_next = merge_next_node(next);
		}

		//next is now new_next
		return next;

	}


	__device__ header * lock_surrounding(){

		header * first;

		while (true){

			header * second = find_next_free();

			//already locked no need
			//second->stall_lock();

			first = second->get_prev();

			if (first->lock()){
				return first;
			}

			second->unlock();

		}

	}

	__device__ static void free(void * address){

		address -= 32;

		header * head = (header * ) address;


		//declare
		head->lock();

		//header * next = head->find_next_free();

		//header * prev = next->get_prev();

		header * prev = head->lock_surrounding();

		header * next = prev->get_next();

		next->stall_lock();

		//At this point
		//head
		//prev
		//next
		// are all locked


		//we can't merge into head nodes that's illegal

		// if (head->get_next_disallocated() == next && !next->check_endpoint()){

		// 	//merge head and next
		// 	header * next_next = next->get_next();

		// 	next_next->lock();

		// 	//merge

		// 	next->merge();

		// 	next = next_next;

		// }

		next = header->merge_all_left(next);


		//at this point, we have prev -> head -> next
		//all locked


		//
		if (prev->get_next_disallocated() == head && !prev->check_endpoint()){

			head->merge();

			//now we have prev_next

		} else {

			//or prev -> next -> head;
			head->unlock();
		}

		//finish unlock

		prev->unlock();
		next->unlock();

	}


};




// struct memory_ptr{

// 	memory_ptr * next;
// 	header_bitfield lock_and_size;


// 	//spin lock waiting for acquisition
// 	//secures acquisition if lock is free to be acquired
// 	//otherwise becomes read-only :D
// 	//on in use-segments lock is not required as they can only be modified by a free
// 	__device__ inline bool lock(){
	
// 	uint16_t result = atomicCAS(&lock_and_size.lock, 0, 1);


// 	while (result != 0){

// 		if (result == 2) return false;

// 		result = atomicCAS(&lock_and_size.lock, 0, 1);
// 	}

// 	//if (result == 0) return true;


// 	return false;

// 	}


// 	__device__ inline bool unlock(){


// 		uint16_t result = atomicCAS(&lock_and_size.lock, 1, 0);

// 		if (result == 1) return true;

// 		return false;

// 	}

// 	__device__ inline void convert_to_used(){

// 		uint64_t result = atomicCAS(&lock_and_size.lock, 1, 2);

// 		#if DEBUG_ASSERTS

// 		assert(result == 1);

// 		#endif

// 	}

// 	__device__ inline void convert_to_locked(){

// 		uint64_t result = atomicCAS(&lock_and_size.lock, 2, 1);

// 		#if DEBUG_ASSERTS

// 		assert(result == 2);

// 		#endif

// 	}

// 	__device__ bool replace_bitfield(header_bitfield old, header_bitfield replacement){


// 		if (atomicCAS((unsigned long long int *)&lock_and_size, old.as_uint, replacement.as_uint) == old.as_uint){
// 			return true;
// 		}

// 		return false;

// 	}

// 	__device__ inline void unlock_and_replace(uint64_t old_size, uint64_t new_size){


// 		header_bitfield old;
// 		old.as_bitfield.lock = 1;
// 		old.as_bitfield.size = old_size;

// 		header_bitfield new_bitfield;
// 		new_bitfield.as_bitfield.lock = 0;
// 		new_bitfield.as_bitfield.size = new_size;

// 		replace_bitfield(old, new_bitfield);

// 		return;

		
		
// 	}

// 	//__device__ inline memory_ptr * atomic_load_next

// 	__device__ inline memory_ptr * atomic_get_next(){

// 		lock();

// 		__threadfence();

// 		memory_ptr * to_return = next;

// 		unlock();

// 		return to_return;

// 	}

// 	__device__ inline memory_ptr * atomic_get_prev(){

// 		lock();

// 		__threadfence();

// 		memory_ptr * to_return = next;

// 		unlock();

// 		return to_return;

// 	}

// 	__device__ inline memory_ptr * get_footer(){

// 		lock();
// 		__threadfence();

// 		char * correct_address = ((char *) this) + 16 + lock.as_bitfield.size;

// 		memory_ptr * footer = (memory_ptr * ) correct_address;

// 		unlock();

// 		return footer;

// 	}

// 	__device__ inline memory_ptr * get_header(){

// 		lock();
// 		__threadfence();

// 		char * correct_address = ((char *) this) - (16 + lock.as_bitfield.size);

// 		memory_ptr * header = (memory_ptr * ) correct_address;

// 		unlock()

// 		return header;

// 	}


// 	//quick debug test - does my footer send me as the header?
// 	__device__ void test_header(){


// 		memory_ptr * footer = get_footer();

// 		assert(footer->get_header() == this);
// 		return;

// 	}

// 	__device__ void test_footer(){


// 		memory_ptr * header = get_header();

// 		assert(header->get_footer() == this);
// 		return;

// 	}


// };

// struct heap_wrapper {

// 	memory_ptr * header;

// 	memory_ptr * footer;

// 	heap_wrapper(){}

// 	static bool is_aligned(void * address){

// 		uint64_t address_as_uint = (uint64_t) address;

// 		return !(address % 16);
// 	}


// 	__device__ void set_next(heap_wrapper next){

// 		header->next = next.header;

// 	}

// 	__device__ void set_prev(heap_wrapper prev){

// 		footer->next = prev.header;

// 	}

// 	static heap_wrapper create_new_node(void * address, uint64_t num_bytes){

// 		//assert the address is 16 byte aligned
// 		#if DEBUG_ASSERTS

// 		assert(heap_wrapper::is_aligned(address));
// 		assert(num_bytes % 16 == 0);
// 		assert(num_bytes > 0);

// 		#endif



// 		uint64_t bytes_inside = num_bytes-32;

// 		heap_wrapper new_heap_wrapper;

// 		heap_wrapper.header = (memory_ptr *) address;

// 		heap_wrapper.header->lock_and_size.as_bitfield.size = bytes_inside;
// 		heap_wrapper.header->lock_and_size.lock = 0;


// 		heap_wrapper.footer = (memory_ptr *) (address + num_bytes - 16);

// 		heap_wrapper.footer->lock_and_size.as_bitfield.size = bytes_inside;
// 		heap_wrapper.footer->lock_and_size.lock = 0;

// 		return heap_wrapper;



// 	}


// 	static heap_wrapper initialize_heap(void * address, uint64_t num_bytes){

// 		dummy_heap_start = heap_wrapper::create_new_node(address, 32);

// 		main_heap_node = heap_wrapper::create_new_node(address+32, num_bytes-64);

// 		dummy_heap_end = heap_wrapper::create_new_node(address+num_bytes-32, 32);

// 		dummy_heap_start.set_next(main_heap_node);
// 		dummy_heap_start.set_prev(dummy_heap_end);


// 		main_heap_node.set_next(dummy_heap_end);
// 		main_heap_node.set_prev(dummy_heap_start);


// 		dummy_heap_end.set_next(dummy_heap_start);
// 		dummy_heap_end.set_prev(main_heap_node);

// 		return main_heap_node;

// 	}


// 	//header * next_node()

// 	__device__ inline heap_wrapper get_next(){

// 		memory_ptr * next_header = header->atomic_get_next();

// 		memory_ptr * next_footer = next_header->get_footer();

// 		#if DEBUG_ASSERTS

// 		next_header->test_header();
// 		next_footer->test_footer();

// 		#endif

// 		heap_wrapper new_heap_wrapper;

// 		new_heap_wrapper.header = next_header;
// 		new_heap_wrapper.footer = next_footer;

// 		return new_heap_wrapper;

// 	}

// 	__device__ bool is_valid(){

// 		if (header->lock()){

// 			header->unlock();
// 			return true;

// 		}

// 		return false;

// 	}

// 	__device__ inline heap_wrapper get_next_from_free(){

// 		memory_ptr * next_header = footer + 1;

// 		memory_ptr * next_footer = next_header->get_footer();

// 		heap_wrapper new_heap_wrapper;

// 		new_heap_wrapper.header = next_header;
// 		new_heap_wrapper.footer = next_footer;

// 		return new_heap_wrapper;

// 	}


// 	__device__ inline heap_wrapper get_prev_from_free(){

// 		memory_ptr * next_footer = header-1;

// 		memory_ptr * next_header = next_footer->get_header();

// 		heap_wrapper new_heap_wrapper;

// 		new_heap_wrapper.header = next_header;
// 		new_heap_wrapper.footer = next_footer;


// 	}


// 	//find the next node that is free in the list and resembles me
// 	//have to check that I won't scan outside of the bounds
// 	static inline heap_wrapper find_prev_free(){

// 		if (footer->next >= footer){

// 			//wrapping around!

// 		}


// 	}

// 	static inline void free_node(void * allocation){

// 		//step back to find the header
// 		memory_ptr * header = (memory_ptr *) (allocation-16);

// 		memory_ptr * footer = header->get_footer();

// 		//swap this node back to a locked state
// 		header->convert_to_locked();

// 		if (footer->next < footer) 

// 	}


// 	link_new_node(heap_wrapper new_node){




// 	}


// }


// //heap pointers
// //have a header
// //	lock variables
// //  pointer to next
// //  size in bytes

// //a footer
// 	// pointer to previous (if free)
// 	//size in bytes


// struct heap_header {


// 	uint lock

// }


// //alignas(Recursive_size<(sizeof(Key) + sizeof(Val))>::result) 
// struct  global_heap_ptr {

// 	public:


// 		//32 byte headers for "unallocated" regions
// 		global_heap_ptr * node_ahead;
// 		global_heap_ptr * node_behind;
// 		uint64_t bytes_available;		
// 		large_unioned_bitfield counters;

// 		//base_heap_ptr * free_list;

		

// 		__host__ __device__ global_heap_ptr(){};

// 		__device__ large_unioned_bitfield atomic_load(){

// 			large_unioned_bitfield ret_field;
// 			ret_field.as_uint = atomicCAS((unsigned long long int *)&counters, 0, 0);
// 			return ret_field;


// 		}

// 		//attempt to swap values in the list
// 		//this can be used to acquire a lock
// 		//or signal a change in status
// 		__device__ bool atomic_swap(large_unioned_bitfield current, large_unioned_bitfield replace){

// 			if (current.as_uint == atomicCAS((unsigned long long int *)&counters, current.as_uint, replace.as_uint)){

// 				return true;
// 			}

// 			return false;

// 		}

// 		//get the offset to the next aligned region
// 		__device__ uint64_t offset_to_next_aligned_region(uint64_t alignment_width){

// 			uint64_t this_as_bits = (uint64_t) this;

// 			//this as bits - (this_as bits % aligment) is how far off alignment we are
// 			// aligment - ^quantity is how many bytes to the next correct region

// 			uint64_t offset = alignment_width - (this_as_bits - (this_as_bits % alignment_width));

// 			return offset;


// 		}

// 		__device__ bool can_request_from_block(uint64_t num_bytes){

// 			if (num_bytes+8 < bytes_available) return true;

// 			return false;

// 		}


// 		//assert that the node exists in a valid state, stall until then
// 		__device__ large_unioned_bitfield get_node_in_valid_state(){

// 			large_unioned_bitfield status = atomic_load();

// 			while(!status.assert_valid()){
// 				status = atomic_load;
// 			}

// 			return status;

// 		}

// 		__device__ large_unioned_bitfield lock_node_in_exclusive_state(){



// 			large_unioned_bitfield status = get_node_in_valid_state();
// 			large_unioned_bitfield new_status = status;

// 			new_status.as_bitfield.first+=1;

// 			while (!atomic_swap(status, new_status)){
// 				status = atomic_load;

// 				while (!status.assert_valid){
// 					status = atomic_load();
// 				}

// 				new_status = status;
// 				new_status.as_bitfield.first+=1;

// 			}


// 			//new status is the current (invalid) state
// 			return new_status;


// 		}

// 		__device__ large_unioned_bitfield lock_node_in_exclusive_state(large_unioned_bitfield old_status){



// 			large_unioned_bitfield status = old_status;
// 			large_unioned_bitfield new_status = status;

// 			new_status.as_bitfield.first+=1;

// 			while (!atomic_swap(status, new_status)){
// 				status = atomic_load;

// 				while (!status.assert_valid){
// 					status = atomic_load();
// 				}

// 				new_status = status;
// 				new_status.as_bitfield.first+=1;

// 			}


// 			//new status is the current (invalid) state
// 			return new_status;


// 		}

// 		__device__ void unlock_node_from_status(large_unioned_bitfield locked_status){


// 			large_unioned_bitfield new_status = locked_status;

// 			new_status.as_bitfield.second += 1;
// 			bool success = atomic_swap(new_status, locked_status);


// 			//this should always succeed as no one can generate a new valid state from 
// 			// our locked state
// 			#if DEBUG_ASSERTS

// 			assert(success);

// 			#endif

// 			return;

// 		}

// 		__device__ void * request_allocation_from_block(uint64_t num_bytes){

// 			//first check just to assert node is valid
// 			large_unioned_bitfield status = get_node_in_valid_state();

// 			//now we can assert that the state is valid
// 			//read number of bytes available
// 			//uint64_t old_num_bytes = bytes_available;

// 			//can't get blood from a tree
// 			//if it can't hold the item return nullptr
// 			if (!can_request_from_block(num_bytes)){
// 				return nullptr;
// 			}

// 			//We think we can request from this node, grab it

// 			large_unioned_bitfield locked_status = lock_node_in_exclusive_state(status);

// 			//we've grabbed the node, but at what cost?

// 			if (!can_request_from_block(num_bytes)){

// 				//failure, swap back to stable state

// 				unlock_node_from_status(locked_status);

// 				return nullptr;

// 			}

// 			//otherwise we can request! figure out how many bytes to detach
// 			uint64_t new_width = bytes_available 



// 		}

// 		//valid from any pointer
// 		__device__ void * request_allocation(uint64_t num_bytes){


// 			//first step is to check if this node has enough space
// 			global_heap_ptr * head = this;


// 			while (global_heap_ptr != head){

// 			}

// 			return false;




// 		}

// 		__device__ void * request_aligned_allocation(uint64_t num_bytes, uint64_t alignment_width){

// 		}

// 		__host__ global_heap_ptr * generate_on_device(uint64_t num_bytes){

// 			void * byte_space;

// 			cudaMalloc((void **)&byte_space, num_bytes);

			

// 			global_heap_ptr host_heap;

// 			host_heap.node_ahead = byte_space;
// 			host_heap.node_behind = byte_space;


// 			global_heap * dev_heap;

// 			cudaMemcpy(dev_heap, &host_heap, sizeof(global_heap), cudaMemcpyHostToDevice);

// 		}


// 		__host__ free_on_device(global_heap * dev_heap){

// 			global_heap host_heap;

// 			cudaMemcpy(&host_heap, dev_heap, sizeof(global_heap), cudaMemcpyDeviceToHost);

// 			base_heap_ptr::free_on_device(host_heap.free_list);


// 		}

		

// };



}

}


#endif //GPU_BLOCK_