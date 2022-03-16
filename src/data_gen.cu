/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>


//#include "include/sorted_block_vqf.cuh"
//#include "include/metadata.cuh"

#include <openssl/rand.h>





__host__ uint64_t * generate_data(uint64_t nitems){


	//malloc space

	uint64_t * vals = (uint64_t *) malloc(nitems * sizeof(uint64_t));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(uint64_t));



		to_fill += togen;

		printf("Generated %llu/%llu\n", to_fill, nitems);
		fflush(stdout);

	}

	return vals;
}

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);


	uint64_t nitems = (1ULL << nbits);

	uint64_t * vals;



	vals = generate_data(nitems);


	//uint64_t * fp_vals;

//	fp_vals = generate_data(nitems);


	printf("Setup done\n");
	fflush(stdout);


	uint64_t char_nitems = nitems * (sizeof(uint64_t));


	char * char_vals = (char *) vals;

	//char * char_fp = (char *) fp_vals;

	//open and write based on name

	char main_str_holder[300];
	//char fp_str_holder[300];


	char main_filename[] = "-data.txt";

	//char fp_filename[] = "-fp.txt";


	strcpy(main_str_holder, argv[2]);

	strcat(main_str_holder, "-");

	strcat(main_str_holder, argv[1]);
	strcat(main_str_holder, main_filename);


	// strcpy(fp_str_holder, argv[2]);

	// strcat(fp_str_holder, "-");

	// strcat(fp_str_holder, argv[1]);
	// strcat(fp_str_holder, fp_filename);




	printf("Main file %s\n", main_str_holder);

//	printf("Alt file %s\n", fp_str_holder);


	std::ofstream vals_file (main_str_holder);

	//std::ofstream fp_file (fp_str_holder);

	//FILE * fp_file = fopen(strcat(argv[1],fp_filename), "w");

	if (vals_file.is_open()){



		for (int i = 0; i < char_nitems; i++){

			vals_file << char_vals[i];


		}

		vals_file << std::endl;


		//int return_val = fputs(char_vals, )

		vals_file.close();

	}


	// if (fp_file.is_open()){

	// 	for (int i = 0; i < char_nitems; i++){

	// 	fp_file << char_fp[i];


	// }

	// fp_file << std::endl;


	// fp_file.close();



	//}




	return 0;

}
