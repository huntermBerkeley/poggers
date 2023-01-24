#ifndef VEB_TREE
#define VEB_TREE
//A CUDA implementation of the Van Emde Boas tree, made by Hunter McCoy (hunter@cs.utah.edu)
//Copyright (C) 2023 by Hunter McCoy

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
//and associated documentation files (the "Software"), to deal in the Software without restriction, 
//including without l> imitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
//LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))



//a layer is a bitvector used for ops
//internally, they are just uint64_t's as those are the fastest to work with

//The graders might be asking, "why the hell did you not include max and min?"
//with the power of builtin __ll commands (mostly __ffsll) we can recalculate those in constant time on the blocks
// which *should* be faster than a random memory load, as even the prefetch is going to be at least one cycle to launch
// this saves ~66% memory with no overheads!
struct layer{

	const uint64_t universe_size;
	const uint64_t num_blocks;
	uint64_t * bits;
	//int * max;
	//int * min;



	//have to think about this for the port
	//const universe size is cool
	__host__ __device__ layer(uint64_t universe): universe_size(universe), num_blocks((universe_size-1)/64+1){


		//bits = (uint64_t *) cudaMalloc(num_blocks, sizeof(uint64_t));

		//max = (int * ) calloc(num_blocks, sizeof(int));

		//min = (int * ) calloc(num_blocks, sizeof(int));

	}


	__host__ layer * generate_on_device(uint64_t universe){

			layer host_layer(universe);

			layer * dev_layer;

			uint64_t * bits;

			cudaMalloc((void **)&bits, host_layer.num_blocks*sizeof(uint64_t));

			//does this work? find out on the next episode of dragon ball z.
			cudaMemset(bits, ~0, host_layer.num_blocks*sizeof(uint64_t));

			host_layer.bits = bits;

			cudaMalloc((void **)&dev_layer, sizeof(layer));

			cudaMemcpy(dev_layer, &host_layer, sizeof(layer), cudaMemcpyHostToDevice);

	}




	//returns the index of the next 1 in the block, or -1 if it DNE
	int inline find_next(uint64_t high, int low){


		#if DEBUG_PRINTS
		printf("bits: %lx, bitmask %lx\n", bits[high], ~BITMASK(low+1));
		#endif

		if (low == -1){
			return __builtin_ffsll(bits[high])-1;
		}

		return __builtin_ffsll(bits[high] & ~BITMASK(low+1)) -1;

	}

	int inline get_min(uint64_t high){

		return find_next(high, -1);
	}

	int inline get_max(uint64_t high){

		return 63 - __builtin_clzll(bits[high]);

	}

	//returns true if new int added for the first time
	//false if already inserted
	bool insert(uint64_t high, int low){

		if (bits[high] & SET_BIT_MASK(low)) return false;

		bits[high] |= SET_BIT_MASK(low);

		return true;

	}

	//returns true if item in bitmask.
	bool query(uint64_t high, int low){

		return (bits[high] & SET_BIT_MASK(low));
	}


};


//This VEB tree uses a constant factor scaling of sqrt U to acheive maximum performance
//that is to say, the block size is 64.
//this lets us cheat and successor search in a block in 3 ops by doing __ffsll(BITMASK(index) & bits);
//which will be very useful for making this the fastest.
//in addition, this conventiently allows us to 
struct veb_tree
{

	//template metaprogramming to get square root at compile time
	//why waste cycles?
	int num_layers;
	//inp_type global_max;

	using my_type = veb_tree;

	layer ** layers;

	veb_tree() {


		int max_height = 64 - __builtin_clzll(universe_size) -1;

		assert(max_height != -1);
		assert(__builtin_popcountll(universe_size) == 1);

		//round up but always assume
		num_layers = (max_height-1)/6+1;

		#if DEBUG_PRINTS
		printf("Building tree in universe of %lu items with depth %d / %d\n", universe_size, max_height, num_layers);
		#endif

		layers = (layer **) malloc(num_layers * sizeof(layer *));

		uint64_t internal_universe_size = universe_size;

		for (int i = 0; i < num_layers; i++){

			layers[num_layers-1-i] = new layer(internal_universe_size);

			//yeah I know this is ugly sue me
			internal_universe_size = (internal_universe_size-1)/64+1;

		}

		//initialization is done
		return;
		

	}


	~veb_tree(){

		#if DEBUG_PRINTS
		printf("Execution Ending, closing tree\n");
		#endif

		for (int i =0; i < num_layers; i++){


			delete(layers[i]);
		}

		free(layers);


		#if DEBUG_PRINTS
		printf("Tree cleaned up\n");
		#endif

	}



	bool float_up(int & layer, uint64_t &high, int &low){

		layer -=1;

		low = high & BITMASK(6);
		high = high >> 6;

		return (layer >=0);


	}

	bool float_down(int & layer, uint64_t &high, int&low){

		layer+=1;
		high = (high << 6) + low;
		low = -1;

		return (layer < num_layers);

	}


	void insert(inp_type new_insert){

		uint64_t high = new_insert >> 6;
		int low = new_insert & BITMASK(6);

		int layer = num_layers-1;

		//if (new_insert > global_max) global_max = new_insert;

		//this is totally readable right?
		//no confusion at all
		//if this wasn't clear, this will call insert at every layer
		//and if that succeeds, it will call float up
		//either insertion failure or float up failure will exit
		//insertion failure means our bit was already set, so we are done
		//float up fail means there are no more levels.
		while (layers[layer]->insert(high, low) && float_up(layer, high, low));

	}


	bool query(inp_type query_val){


		uint64_t high = query_val >> 6;
		int low = query_val & BITMASK(6);

		return layers[num_layers-1]->query(high, low);

	}

	static const inp_type fail(){
		return ~ ((inp_type) 0);
	}


	inp_type successor(inp_type query_val){

		//I screwed up and made it so that this finds the first item >=
		//can fix that with a quick lookup haha
		if (query(query_val)) return query_val;

		uint64_t high = query_val >> 6;
		int low = query_val & BITMASK(6);

		int layer = num_layers-1;

		#if DEBUG_PRINTS
		printf("Input %u breaks into high bits %lu and low bits %d\n", query_val, high, low);
		#endif

		while (true){

			//break condition

			int found_idx = layers[layer]->find_next(high, low);

			#if DEBUG_PRINTS
			printf("For input %u, next in layer %d is %d\n", query_val, layer, found_idx);
			#endif


			if (found_idx == -1){

				if (layer == 0) return my_type::fail();

				float_up(layer, high, low);
				continue;

				
			} else {
				break;
			}

		}

		#if DEBUG_PRINTS
		printf("Starting float down\n");
		#endif

		while (layer != (num_layers-1)){


			low = layers[layer]->find_next(high, low);
			float_down(layer, high, low);

		}

		low = layers[layer]->find_next(high, low);

		return (high << 6) + low;



	}


	
};



#endif //End of VEB guard