#ifndef SORTING_HELPER_H
#define SORTING_HELPER_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/sorting_helper.cuh"


//A collection of warp level sorting tools and verification
// These don't need to be integrated into the main block so I'm clipping them for faster debugging


__device__ void swap(uint64_t * items, int i, int j);


__device__ int greatest_power_of_two(int n);

__device__ void compare_and_swap(uint64_t * items, int i, int j, bool dir);

__device__ void bitonicMerge(uint64_t * items, int low, int count, bool dir, int warpID);

__device__ void bitonicSort(uint64_t * items, int low, int count, bool dir, int warpID);


__host__ __device__ bool byte_assert_sorted(uint64_t * items, uint64_t nitems);

__host__ __device__ bool two_byte_assert_sorted(uint64_t * items, uint64_t nitems);


__device__ void byteBitonicSort(uint64_t * items, int low, int count, bool dir, int warpID);

__device__ void byteBitonicMerge(uint64_t * items, int low, int count, bool dir, int warpID);

__host__ __device__ bool short_byte_assert_sorted(uint8_t * items, uint64_t nitems);

__host__ __device__ bool sixteen_byte_assert_sorted(uint16_t * items, uint64_t nitems);

__device__ void shortByteBitonicSort(uint8_t * items, int low, int count, bool dir, int warpID);

__device__ void bubble_sort(uint8_t * tags, int fill, int warpID);

__device__ void big_bubble_sort(uint64_t * tags, int fill, int warpID);

__device__ void short_warp_sort(uint8_t * items, int nitems, int teamID, int warpID);


__device__ void merge_dual_arrays(uint8_t * temp_tags, uint8_t * primary, uint8_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID);

__device__ void merge_dual_arrays_8_bit_64_bit(uint8_t * temp_tags, uint8_t * primary, uint64_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID);

__device__ void merge_dual_arrays_16_bit_64_bit(uint16_t * temp_tags, uint16_t * primary, uint64_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID);

__device__ void merge_dual_arrays_sixteen(uint16_t * temp_tags, uint16_t * primary, uint16_t * secondary, int primary_nitems, int secondary_nitems, int teamID, int warpID);


//triplicate array merges
__device__ void merge_3_into_tags_16_bit(uint16_t * tags, uint64_t * global_buffer, int buffer_count, uint16_t * primary, int primary_nitems, uint16_t * secondary, int secondary_nitems, int teamID, int warpID);




__device__ void insertion_sort_max(uint8_t * items, int nitems);

__device__ void insertion_sort_16(uint16_t * items, int nitems);

__device__ void sorting_network_8_bit(uint8_t * items, int nitems, int warpID);

__device__ void sorting_network_16_bit(uint16_t * items, int nitems, int warpID);



#endif