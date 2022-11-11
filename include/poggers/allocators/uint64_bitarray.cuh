#ifndef POGGERS_UINT64_BITARRAY
#define POGGERS_UINT64_BITARRAY


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include "stdio.h"
#include "assert.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>


namespace cg = cooperative_groups;


//selects 01 from every pair
#define FULL_BIT_MASK 0x5555555555555555ULL;

//selects 10 from every pair
#define HAS_CHILD_BIT_MASK 0xAAAAAAAAAAAAAAAAULL;



//a pointer list managing a set section of device memory
namespace poggers {


namespace allocators { 



struct uint64_t_bitarr {


	uint64_t bits;

	__host__ __device__ uint64_t_bitarr(){
		bits = 0ULL;
	}

	__host__ __device__ uint64_t_bitarr(uint64_t ext_bits){
		bits = ext_bits;
	}

	__device__ int get_random_active_bit(){

		uint64_t tid = threadIdx.x*threadIdx.x;

		//big ol prime *1610612741ULL
		int random_cutoff = ((tid*1610612741ULL) % 64);

		//does this mask need to check on 64 bit case?
		//no actually cause there is no functional difference as its just the last bit?

		uint64_t random_mask = ((1ULL << random_cutoff) -1);

		int valid_upper = __ffsll(bits & (~random_mask)) -1;

		if (valid_upper != -1){
			return valid_upper;
		}

		//upper bits are not set (from above) so we can save an op for __ffsll and find first set for whole thing.
		return __ffsll(bits) -1;



		// uint64_t random_mask = ((1ULL << random_cutoff) -1);

		// int valid_upper = __ffsll(bits & (~random_mask)) -1;

		// if (valid_upper != -1){
		// 	return valid_upper;
		// }

		// //upper bits are not set (from above) so we can save an op for __ffsll and find first set for whole thing.
		// return __ffsll(bits) -1;


	}

	__device__ int get_first_active_bit(){

		return __ffsll(bits) -1;
	}

	__device__ void invert(){
		bits = ~bits;
	}

	__device__ inline uint64_t generate_set_mask(int index){

		return (1ULL) << index;

	}

	__device__ inline uint64_t generate_unset_mask(int index){

		return ~generate_set_mask(index);

	}

	__device__ bool set_bit_atomic(int index){

		uint64_t set_mask = generate_set_mask(index);

		uint64_t old = atomicOr((unsigned long long int *) this, set_mask);

		//old should be empty

		return (~old & set_mask);

	}

	__device__ bool unset_bit_atomic(int index){

		uint64_t unset_mask = generate_unset_mask(index);

		uint64_t old = atomicAnd((unsigned long long int * ) this, unset_mask);

		return (old & ~unset_mask);

	}

	__device__ uint64_t_bitarr global_load_this(){

		return (uint64_t_bitarr) poggers::utils::ldca((uint64_t *) this);

	}

	__device__ int get_fill(){
		return __popcll(bits);
	}

	__device__ uint64_t_bitarr swap_to_empty(){

		return (uint64_t_bitarr) atomicExch((unsigned long long int *) this, 0ULL);

	}

	__device__ bool set_bits(uint64_t ext_bits){

		return (atomicCAS( (unsigned long long int *) this, 0ULL, (unsigned long long int) ext_bits) == 0ULL);
	}

	__device__ uint64_t get_bits(){
		return bits;
	}

	__device__ int bits_before_index(int index){

		if (index == 63){

			return __popcll(bits);
		}

		uint64_t mask = (1ULL <<index);

		return __popcll(bits & mask);

	}

	__device__ void apply_mask(uint64_t mask){

		bits = bits & mask;
	}

	__host__ __device__ operator uint64_t() const { return bits; }

};


struct warp_lock {

	uint64_t_bitarr lock_bits;

	__device__ void init(){

		lock_bits = 0ULL;

	}

	__device__ int get_warp_bit(){

		return (threadIdx.x / 32);

	}

	__device__ bool lock(){

		return lock_bits.set_bit_atomic(get_warp_bit());

	}

	__device__ void unlock(){

		lock_bits.unset_bit_atomic(get_warp_bit());

	}

	__device__ void spin_lock(){


		while (!lock());

	}

};



struct alloc_bitarr{

	void * memmap;
	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];

	__device__ void init(){

		manager_bits.bits = ~(0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = ~(0ULL);
		}
		//at some point work on this
		memmap = nullptr;

	}


	__device__ void attach_allocation(void * ext_alloc){

		memmap = ext_alloc;

	}

	//request one item for this thread
	__device__ bool bit_malloc(void * & allocation, uint64_t & remainder, void * & remainder_offset, bool & is_leader){


		//group
		//

		//team shares the load
		uint64_t_bitarr local_copy = manager_bits.global_load_this();


		#if DEBUG_PRINTS
		cg::coalesced_group active_threads = cg::coalesced_threads();

		if (active_threads.thread_rank() == 0){
			printf("%d/%d %llx\n", active_threads.thread_rank(), active_threads.size(), local_copy);
		}
		#endif
		

		while(local_copy.get_fill() != 0){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			int allocation_index_bit = 0;

			if (active_threads.thread_rank() == 0){

				//allocation_index_bit = local_copy.get_first_active_bit();

				allocation_index_bit = local_copy.get_random_active_bit();

			}
			
			allocation_index_bit = active_threads.shfl(allocation_index_bit, 0);
			

			uint64_t_bitarr ext_bits;

			bool ballot_bit_set = false;

			if (active_threads.thread_rank() == 0){


				if (manager_bits.unset_bit_atomic(allocation_index_bit)){


					ext_bits = alloc_bits[allocation_index_bit].swap_to_empty();

					ballot_bit_set = true;



				}


			}

			//at this point, ballot_bit_set and ext_bits are set in thread 0
			//so we ballot on if we can leave the loop

			if (active_threads.ballot(ballot_bit_set)){


				 
				ext_bits = active_threads.shfl(ext_bits, 0);

				#if DEBUG_PRINTS
				if (active_threads.thread_rank() == 0){
					printf("%d/%d sees ext_bits for %d as %llx\n", active_threads.thread_rank(), active_threads.size(), allocation_index_bit, ext_bits);
				}
				#endif


				if (active_threads.thread_rank()+1 <= ext_bits.get_fill()){

					//next step: gather threads
					cg::coalesced_group coalesced_threads = cg::coalesced_threads();

					#if DEBUG_PRINTS
					if (coalesced_threads.thread_rank() == 0){
						printf("Leader is %d, sees %d threads coalesced.\n", active_threads.thread_rank(), coalesced_threads.size());
					}
					#endif


					//how to sync outputs?
					//everyone should pick a random lane?

					//how to coalesce after lanes are picked


					//options
					//1) grab an allocation of the first n and try to  
					//2) select the first n bits ahead of time.

					//int bits_needed =  (ext_bits.get_fill() - active_threads.size());

					//int my_bits = bits_before_index(active_threads.thread_rank());

					// bool ballot = (bits_needeed == my_bits);

					// int result = coalesced_threads.ballot(ballot);

					
					int my_index;

					while (true){

						cg::coalesced_group searching_group = cg::coalesced_threads();

						my_index = ext_bits.get_random_active_bit();

						#if DEBUG_PRINTS
						if (searching_group.thread_rank() == 0){
							printf("Leader is %d/%d, sees ext bits as %llx\n", coalesced_threads.thread_rank(), searching_group.size(), ext_bits);
						}
						#endif

						//any threads still searching group together
						//do an exclusive scan on the OR bits 

						//if the exclusive OR result doesn't contain your bit you are free to modify!

						//last thread knows the true state of the system, so broadcast changes.

						

						uint64_t my_mask = (1ULL) << my_index;

						//now scan across the masks
						uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

						//final thread needs to broadcast updates
						if (searching_group.thread_rank() == searching_group.size()-1){

							//doesn't matter as the scan only adds bits
							//not to set the mask to all bits not taken
							uint64_t final_mask = ~(scanned_mask | my_mask);

							ext_bits.apply_mask(final_mask);

						}

						//everyone now has an updated final copy of ext bits?
						ext_bits = searching_group.shfl(ext_bits, searching_group.size()-1);


						if (!(scanned_mask & my_mask)){

							//I received an item!
							//allocation has already been marked and index is set
							//break to recoalesce for exit
							break;



						}


					} //internal while loop

					coalesced_threads.sync();

					//TODO - take offset based on alloc size
					//for now these are one byte allocs
					allocation = (void *) (memmap + my_index + 64*allocation_index_bit);

					//someone now has the minimum.
					int my_fill = ext_bits.get_fill();

					int lowest_fill = cg::reduce(coalesced_threads, my_fill, cg::less<int>());

					int leader = __ffs(coalesced_threads.ballot(lowest_fill == my_fill))-1;

					#if DEBUG_PRINTS
					if (leader == coalesced_threads.thread_rank()){
						printf("Leader reports lowest fill: %d, my_fill: %d, bits: %llx\n", lowest_fill, my_fill, ext_bits);
					}
					#endif
					//printf("Leader is %d\n", leader, coalesced_threads.size());

					if ((leader == coalesced_threads.thread_rank())){

						is_leader = true;
						remainder = ext_bits;

						remainder_offset = memmap + 64*allocation_index_bit;

					} else {

						is_leader = false;
						remainder = 0;
						remainder_offset = nullptr;

					}


					return true;




				} //if active alloc


			} //if bit set

			


			//one extra inserted above this
			//on failure reload local copy
			local_copy = manager_bits.global_load_this();

			} //current end of while loop?

		return false;	

	}

	




};



//Correctness precondition
//0000000000000000 is empty key
//if you create it you *will* destroy it
//so other threads don't touch blocks that show themselves as 0ULL
//This allows it to act as the intermediate state of blocks
//and allows the remove pipeline to be identical to above ^
//as we first remove and then re-add if there are leftovers.
struct storage_bitmap{


	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];
	void * memmap[64];


	__device__ void init(){

		manager_bits.bits = (0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = (0ULL);
			memmap[i] = nullptr;
		}


		

	}


	__device__ bool attach_buffer(void * ext_buffer, uint64_t ext_bits){


		//group
		//cg::coalesced_group active_threads = cg::coalesced_threads();

		//team shares the load
		uint64_t_bitarr local_copy = manager_bits.global_load_this();

		while (local_copy.get_fill() != 64){

			local_copy.invert();

			#if DEBUG_PRINTS
			printf("Copy: %llx\n", local_copy);
			#endif


				//allocation_index_bit = local_copy.get_first_active_bit();

			int allocation_index_bit = local_copy.get_random_active_bit();


			#if DEBUG_PRINTS
			printf("Bit chosen is %d / %llx, %llx\n", allocation_index_bit, manager_bits, alloc_bits[allocation_index_bit]);
			#endif

			if (alloc_bits[allocation_index_bit].set_bits(ext_bits)){



				if (manager_bits.set_bit_atomic(allocation_index_bit)){

					#if DEBUG_PRINTS
					printf("Manager bit set!\n");
					#endif
				

					return true;

				} else {
					//if you swap out you *must* succeed
					printf("Failure\n");
					assert(1==0);
				}


			}


			local_copy = manager_bits.global_load_this();


		}


		return false;


	}


	__device__ bool bit_malloc(void * & allocation){


		//group
		//cg::coalesced_group active_threads = cg::coalesced_threads();

		//team shares the load
		uint64_t_bitarr local_copy = manager_bits.global_load_this();

		#if DEBUG_PRINTS
		if (active_threads.thread_rank() == 0){
			printf("%d/%d %llx\n", active_threads.thread_rank(), active_threads.size(), local_copy);
		}
		#endif
		

		while(local_copy.get_fill() != 0){

			cg::coalesced_group active_threads = cg::coalesced_threads();

			int allocation_index_bit = 0;

			//does removing this gate affect performance?

			if (active_threads.thread_rank() == 0){

				//allocation_index_bit = local_copy.get_first_active_bit();

				allocation_index_bit = local_copy.get_random_active_bit();

			}
			
			allocation_index_bit = active_threads.shfl(allocation_index_bit, 0);
			

			uint64_t_bitarr ext_bits;

			bool ballot_bit_set = false;

			if (active_threads.thread_rank() == 0){


				if (manager_bits.unset_bit_atomic(allocation_index_bit)){


					ext_bits = alloc_bits[allocation_index_bit].swap_to_empty();

					ballot_bit_set = true;



				}


			}

			//at this point, ballot_bit_set and ext_bits are set in thread 0
			//so we ballot on if we can leave the loop

			if (active_threads.ballot(ballot_bit_set)){


				 
				ext_bits = active_threads.shfl(ext_bits, 0);

				#if DEBUG_PRINTS
				if (active_threads.thread_rank() == 0){
					printf("%d/%d sees ext_bits for %d as %llx\n", active_threads.thread_rank(), active_threads.size(), allocation_index_bit, ext_bits);
				}
				#endif


				if (active_threads.thread_rank()+1 <= ext_bits.get_fill()){

					//next step: gather threads
					cg::coalesced_group coalesced_threads = cg::coalesced_threads();

					#if DEBUG_PRINTS
					if (coalesced_threads.thread_rank() == 0){
						printf("Leader is %d, sees %d threads coalesced.\n", active_threads.thread_rank(), coalesced_threads.size());
					}
					#endif

					//how to sync outputs?
					//everyone should pick a random lane?

					//how to coalesce after lanes are picked


					//options
					//1) grab an allocation of the first n and try to  
					//2) select the first n bits ahead of time.

					//int bits_needed =  (ext_bits.get_fill() - active_threads.size());

					//int my_bits = bits_before_index(active_threads.thread_rank());

					// bool ballot = (bits_needeed == my_bits);

					// int result = coalesced_threads.ballot(ballot);

					
					int my_index;

					while (true){

						cg::coalesced_group searching_group = cg::coalesced_threads();

						my_index = ext_bits.get_random_active_bit();

						#if DEBUG_PRINTS
						if (searching_group.thread_rank() == 0){
							printf("Leader is %d/%d, sees ext bits as %llx\n", coalesced_threads.thread_rank(), searching_group.size(), ext_bits);
						}
						#endif

						//any threads still searching group together
						//do an exclusive scan on the OR bits 

						//if the exclusive OR result doesn't contain your bit you are free to modify!

						//last thread knows the true state of the system, so broadcast changes.

						

						uint64_t my_mask = (1ULL) << my_index;

						//now scan across the masks
						uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

						//final thread needs to broadcast updates
						if (searching_group.thread_rank() == searching_group.size()-1){

							//doesn't matter as the scan only adds bits
							//not to set the mask to all bits not taken
							uint64_t final_mask = ~(scanned_mask | my_mask);

							ext_bits.apply_mask(final_mask);

						}

						//everyone now has an updated final copy of ext bits?
						ext_bits = searching_group.shfl(ext_bits, searching_group.size()-1);


						if (!(scanned_mask & my_mask)){

							//I received an item!
							//allocation has already been marked and index is set
							//break to recoalesce for exit
							break;



						}


					} //internal while loop

					coalesced_threads.sync();

					//TODO - take offset based on alloc size
					//for now these are one byte allocs
					allocation = (void *) (memmap[allocation_index_bit] + my_index);

					//someone now has the minimum.
					int my_fill = ext_bits.get_fill();

					int lowest_fill = cg::reduce(coalesced_threads, my_fill, cg::less<int>());

					int leader = __ffs(coalesced_threads.ballot(lowest_fill == my_fill))-1;

					#if DEBUG_PRINTS
					if (leader == coalesced_threads.thread_rank()){
						printf("Leader reports lowest fill: %d, my_fill: %d, bits: %llx\n", lowest_fill, my_fill, ext_bits);
					}
					#endif
					//printf("Leader is %d\n", leader, coalesced_threads.size());


					if ((ext_bits.get_fill() > 0) && (leader == coalesced_threads.thread_rank())){

						attach_buffer(memmap, ext_bits);

					}

					return true;




				} //if active alloc


			} //if bit set

			


			//one extra inserted above this
			//on failure reload local copy
			local_copy = manager_bits.global_load_this();

			} //current end of while loop?

		return false;	

	}

	




};


__device__ bool alloc_with_locks(void *& allocation, alloc_bitarr * manager, storage_bitmap * block_storage){

	__shared__ warp_lock team_lock;

	while (true){

		cg::coalesced_group grouping = cg::coalesced_threads();

		bool ballot = false;

		if (grouping.thread_rank() == 0){	

			//one thread groups;

			ballot = team_lock.lock();

		}

		if (grouping.ballot(ballot)) break;

	}

	cg::coalesced_group in_lock = cg::coalesced_threads();
	//team has locked
	bool ballot = false;

	if (block_storage->bit_malloc(allocation)){

		ballot = true;
	}

	//if 100% of requests are satisfied, we are all returning, so one thread needs to drop lock.
	if ( __popc(in_lock.ballot(ballot)) == in_lock.size()){

		if (in_lock.thread_rank() == 0){
			team_lock.unlock();
		}

	}

	if (ballot){
		return true;
	}

	//everyone else now can access the main alloc

	uint64_t remainder;
	void * remainder_offset;
	bool is_leader = false;

	bool bit_malloc_result = manager->bit_malloc(allocation, remainder, remainder_offset, is_leader);

	if (is_leader){
	      
	      bool result = block_storage->attach_buffer(remainder_offset, remainder);
	      
	      team_lock.unlock();
	}

	return bit_malloc_result;


}



__global__ void setup_first_level(uint64_t * items, uint64_t num_uints_lowest_level, uint64_t num_items_lowest_level){

	//float through each level
	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= num_uints_lowest_level) return;

	uint64_t my_uint = 0ULL;

	for (int i = 0; i < 32; i++){

		if (tid * 32 + i < num_items_lowest_level){
			my_uint |= (3ULL << (2*i));
		}

	}

	items[tid] = my_uint;

}

//Given a next level, initialize the level above it. This is (AFAIK) fanout agnostic.


//setup


__global__ void setup_next_level(uint64_t * prev_level, uint64_t * next_level, uint64_t num_items_lower_level){


	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	uint64_t my_uint = 0ULL;

	//if this is true you have no children.
	if (tid*32 >= num_items_lower_level){
		return;
	}

	for (int i = 0; i < 32; i++){

		if (tid*32+i < num_items_lower_level){

			uint64_t items = prev_level[tid*32+i];

			if (__popcll(items) == 64){
				my_uint |= (3ULL << (2*i));
			} else {
				my_uint |= (2ULL << (2*i));
			}

		}

	}

	next_level[tid] = my_uint;


}

//the bitbuddy allocator (patent pending)
//uses teams of buddies managed with bitmaps to quickly allocate and deallocate large blocks of memory
//this allocator assigns two bits to every allocation type: one bit for has_valid_children and one bit for allocable
//Together these allow for fast traversal of the tree while still maintaining constant-time allocs once a suitable match has been found.
//The process is this: The size at the top is known, along with a target
//While 


//TODO: Ask prashant about 3rd bit per item! This could handle allocations of scaling size. - Since fanout is 32x, we can build larger-ish allocations by grabbing contiguous segments
//and then mark that those segments are together with a unary counter.


//TODO list:
// 1) Init - every item should contribute bits to layer above it, repeat until only one layer is left
// 2) Malloc - taken from top rec

//Bit ordering
// Children - available  / fully available
//0x3 for full children allocs otherwise 0x2.
//we don't have to worry about other cases on boot.


//bit configurations
// 00 - All children allocated
// 11 - All children free
// 01 - Main item is alloced
// 10 - Some children are alloced


//New ordering - simplifies some ops
// 11 - all available
// 10 - children available
// 00 - alloced as whole node
// 01 - fully alloced

//swap procedure - setting the first bit to 0 means that the state of the system is not configurable by other threads
// this is becuase both alloced as whole and fully alloced are end states.
// so when allocing a node we swap out the first bit to 0 and observe the state
// if it was already 0, we didn't do anything and do not own this node (read failure)
// if it was 1, we might own the node! check if previous state was 10
// if that is the case, roll back to 10... Whoops
// otherwise the node has been successfully alloced

//When rolling up the list we unset the other bit
//if the observed state was 10 or 11, we're done and are good
//if the state was 00 or 01 before we did something wrong and did not really allocate
//this necessitates a rollback of our changes in the lower levels
//and a reset to the original 01 if that was the previous state.

//big change - when allocating a new node, we should originally swap to 01 and disable the node
//then float up
//this allows for 00 nodes to not exist unless the node is explicitly fully alloced
//this maintains that the top 00 found in an items path is the item on free
//so we start at the top and float down until we see a 00. 

struct bitbuddy_allocator {

	uint64_t num_levels;
	//uint64_t top_size;
	int power_at_top;

	int power_at_bottom;

	uint64_t fanout;

	void * memory_segment;

	uint64_t ** levels;


	static __host__ bitbuddy_allocator * generate_on_device(void * ext_memory, uint64_t num_bytes, uint64_t desired_fanout, uint64_t bytes_at_lowest_level){

		//this should correct for it.
		uint64_t num_items_lowest_level = (num_bytes)/bytes_at_lowest_level;

		int ext_power_at_bottom = get_power_of_offset(bytes_at_lowest_level);

		//With the current scheme, we can pack up to 32 items into one bin simultaneously.
		uint64_t num_uints_lowest_level = (num_items_lowest_level -1 )/32 + 1;

		//I think setup is identical regardless of fanout? that's pretty neat. May require special control logic

		std::vector<uint64_t * > ext_levels;

		uint64_t * current_level;

		cudaMalloc((void **)&current_level, sizeof(uint64_t)*num_uintss_lowest_level);

		cudaDeviceSynchronize();

		setup_first_level<<<(num_items_lowest_level -1)/1024 + 1, 1024>>>(current_level);

		ext_levels.push_back(current_level);

		uint64_t ext_bytes_at_top = bytes_at_lowest_level;


		while (num_items_lowest_level > 64){

			//Calculate how much space needs to be reserved.
			uint64_t num_uints_next_level = (num_uints_lowest_level-1)/32+1;

			uint64_t * next_level;

			cudaMalloc((void **)&next_level, sizeof(uint64_t)*num_uints_next_level);

			setup_next_level(current_level, next_level, num_uints_lowest_level);

			cudaDeviceSynchronize();

			num_items_lowest_level = num_uints_next_level;

			ext_levels.push_back(next_level);

			current_level = next_level;

			//size doubles at every iteration.
			ext_bytes_at_top = ext_bytes_at_top*32;

		}

		
		//now that we have all of this, construct the host version

		bitbuddy_allocator * host_version;

		cudaMallocHost((void **)&host_version, sizeof(bitbuddy_allocator));


		uint64_t ** ext_levels_arr;

		cudaMalloc((void **)&ext_levels_arr, sizeof(uint64_t *)*ext_levels.size());

		cudaMemcpy(ext_levels_arr, ext_levels.data(), sizeof(uint64_t *)*ext_levels.size(), cudaMemcpyHostToDevice);

		host_version->num_levels = ext_levels.size();

		host_version->levels = ext_levels_arr;

		host_version->fanout = desired_fanout;

		host_version->power_at_top = convert_to_max_p2(ext_bytes_at_top);

		host_version->memory = ext_memory;


		bitbuddy_allocator * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(bitbuddy_allocator));

		cudaMemcpy(dev_version, host_version, sizeof(bitbuddy_allocator), cudaMemcpyHostToDevice);

		cudaFreeHost(host_version);

		return dev_version;

	__device__ inline uint64_t set_first_bit(uint64_t * address, int index){

	}

	__device__ inline uint64_t set_second_bit(uint64_t * address, int index){

	}

	__device__ inline uint64_t set_first_bit_atomic(uint64_t * address, int index){



	}

	__device__ inline uint64_t set_second_bit_atomic(uint64_t * address, int index){

	}

	__device__ inline uint64_t generate_set_bit_mask(int index){
		
	}

	__device__ inline uint64_t generate_unset_bit_mask(int index){

	}


	//Precondition: Can only be called on valid pointers.
	//if we maintain this, we don't need to check anything
	__device__ uint64_t select_down(uint64_t current_level, int next_id){

		return (current_level*32+next_id);

	}

	__device__ uint64_t select_up(uint64_t current_level){
		//may
		return current_level/32;
	}

	//In malloc, we are looking for a selection of our size that is 11, and we want to convert it to 00
	//do this by setting it to 01 first 
	__device__ void * malloc(uint64_t num_bytes){


		int p2_desired = convert_to_max_p2(num_bytes);

		if (p2_desired > power_at_top || p2_desired < power_at_bottom){
			return nullptr;
		}

		//we can now move down to any level safely! what we're looking for must be in the table
		int current_power = power_at_top;

		uint64_t selection_id = 0;

		//Failure case - loop

		while (true){

			while (current_power != p2_desired){

				//pick a valid randomly;

				int selection = -1;

				volatile int depth = (power_at_top - p2_desired);

				//once we get deep enough, start to partition randomly
				//this ensures high speed
				if (depth > 2){
					selection = find_valid_with_children_random(levels[depth][selection_id]);
				} else {
					selection = find_valid_with_children_first(levels[depth][selection_id]);
				}


				if (selection == -1){
					//this node isn't viable! float up if we can
					//otherwise no nodes available
					if (current_power == power_at_top){
						return nullptr;
					} else {
						current_power -= 1;
						selection_id = select_up(selection_id);
					}
				}

				//otherwise, we have avalid path down!

				current_power +=1;
				selection_id = select_down(selection_id, selection);
				

			}

			//any time we exit this loop we are in a valid area
			//should be at least 1 potential person in this state

			uint64_t selection_bits = levels[depth][selection_id];


			while (selection_bits){
				int selection = find_valid_full_child_first(selection_bits);

				if (selection == -1){
					selection_id = select_up(selection_id);
					break;
				} else {

					//selection is a valid choice!
					//first attempt to grab all bits below
					uint64_t sub_selection = select_down(selection_id, selection);

					//at the lowest depth, there is no need to CAS
					//otherwise we need to assert that 
					if (depth == power_at_bottom || (atomicCAS((unsigned long long int *)&levels[depth+1][sub_selection], (unsigned long long int) ~0ULL, (unsigned long long int)0ULL) == ~0ULL)){

						//we can unset our bits!
						//setting to configuration 01 from 11

						const uint64_t mask = ~(1ULL << (2*selection+1));

						selection_bits = atomicAnd((unsigned long long int *)&levels[depth+1][sub_selection], (unsigned long long int) mask);

						//need to check if 2*selection+1th is set
						if (((selection_bits >> (2*selection)) & 3ULL) == 3ULL){
							//alloc has been given - we think!

							if (float_unset_up()){
								return select_memory_pointer(depth, selection_id, selection);
							}

						} else {
							//this can only occur if the layer above secured a unique allocation.
							//if it does we just float back up
							depth -=1;
							selection = select_up(selection_id);

						}


					}


				}


			}



		}

	}


	__device__ inline int get_power_of_offset(uint64_t bytes){
		__ffsll((byte_offset))-1;
	}

	__device__ void free(void * allocation){

		//To free quickly, we can jump to the largest allocation that this could be
		//and then float down


		uint64_t byte_offset = ((uint64_t) allocation) - ((uint64_t) memory);

		//thought experiment - 2 levels, top level is 000.....01

		int offset = get_power_of_offset(byte_offset);

		assert offset > 

		//bottom level is all 0s

		//64 bytes available
		//so if %64 either top or bottom left
		//else not
		//so start off assert not





	}


	//given a bitarry representing the lower containers
	//find an item that is fully available: code - 11
	//think we can do this with a bitshift?, do an atomicAND over the bits
	//selects the first instance to lower fragmentation.
	__device__ inline int find_valid_full_child_first(uint64_t bits){

		//start with downshift
		uint64_t downshift = bits >> 1;

		uint64_t_bitarr valid_full(((bits & downshift) & FULL_BIT_MASK));

		return valid_full.get_first_active_bit();



	}

	//same as function above, try to detect pairs that are exactly 11
	//but of all active pairs we should return a random selection for throughput
	__device__ inline int find_valid_full_child_random(uint64_t bits){

		//start with downshift
		uint64_t downshift = bits >> 1;

		uint64_t_bitarr valid_full(((bits & downshift) & FULL_BIT_MASK));

		return valid_full.get_random_active_bit();


	}

	//find a child that is 
	//looking for 10 or 11
	//easy enough
	__device__ inline int find_valid_with_children_first(uint64_t bits){

		uint64_t_bitarr valid_bits(bits & HAS_CHILD_BIT_MASK);

		return valid_bits.get_first_active_bit();

	}

	__device__ inline int find_valid_with_children_first(uint64_t bits){

		uint64_t_bitarr valid_bits(bits & HAS_CHILD_BIT_MASK);

		return valid_bits.get_first_active_bit();

	}

	//KISS
	//convert incoming requests to container that can fulfill said requests
	//working with p2s makes the math simple and easy to understand
	__device__ inline int convert_to_max_p2(uint64_t bytes_requested){

		//largest container is one larger than largest bit

		int largest_bit_set = 63 - __ffsll(__brevll(bytes_requested));

		return largest_bit_set


	}



	}



}



}

}


#endif //GPU_BLOCK_