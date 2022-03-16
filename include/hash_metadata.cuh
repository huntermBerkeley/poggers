#ifndef HASH_METADATA
#define HASH_METADATA

#define DEBUG_ASSERTS 0



//number of warps launched per grid block
#define WARPS_PER_BLOCK 16
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)


#define TAG_BITS 64

#define MAX_PROBE 50

#define KEY_EMPTY 0


#endif