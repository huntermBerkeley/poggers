#ifndef METADATA
#define METADATA

#define DEBUG_ASSERTS 0
#define MAX_FILL 28
#define SINGLE_REGION 0
#define FILL_CUTOFF 28

//do blocks assume exclusive access? if yes, no need to lock
//this is useful for batched scenarios.
#define EXCLUSIVE_ACCESS 1


//number of warps launched per grid block
#define WARPS_PER_BLOCK 16
#define BLOCK_SIZE 512

//# of blocks to be inserted per warp in the bulked insert phase
#define REGIONS_PER_WARP 8


//power of 2 metadata
#define POWER_BLOCK_SIZE 1024
#define TOMBSTONE 1000000000000ULL

#endif