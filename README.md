POGGERS: Portable Optimized GPU Generic Structures
__________________________________

POGGERS is a header-only library of high-performnace CUDA data structures.

This library is composed of template objects that allow you to quickly build and modify a modular table while maintaining high performance.

The end goal of the library is to add simple-to-use data structures that simplify the use of CUDA and bring a syntax similar to mainline C++.

For now, the only features that are supported are the hash table and filter components.

FEATURES
________________

1) Hash Tables and Filtering (TCF)
	- A set of high-performance template components for building key-value data strucutures.
	- Modular components to specify data layout, access patterns, and false-positive rate.
  - Recursive template structure lets you build efficient multi-layer tables with minimal effort.



Adding POGGERS
____________________

To add poggers to a project, you can use the [CMake Package Manager](https://github.com/cpm-cmake/CPM.cmake)


To add CPM, add 

```include(cmake/CPM.cmake)``` 

to your cmake file.

To add poggers, include the following snippet and select a version.

If you remove the version tag, CPM will pull the most up-to-date build.

```
CPMAddPackage(
  NAME poggers
  GITHUB_REPOSITORY huntermBerkeley/poggers
  GIT_TAG origin/main
  VERSION v1.0
)
```

To cache the library, specify a download folder:

```set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/downloaded_libraries")```



Building Tests
___________________

There are a series of optional tests that can be included with the build.

To build the tests, specify the ```-DPOGGERS_BUILD_TESTS=ON``` flag to CMake.


Two Choice Filter
_______________________

The Two Choice Filter is a fast approximate data structure based on two-choice-hashing. The base version of the TCF supports insertions, queries, and key-value association. To include the base version of the TCF, include ```<poggers/data_structs/tcf.cuh>``` or add the following template:


```
  using tcf_backing_table = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
  using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tcf_backing_table>;
```


TCF Deletes
_______________________

To guarantee that the TCF does not introduce false negatives during deletion, the hash functions used must be depedent. This can be acheived by using XOR power of two hashing, along with linear probing for the backing table (with sufficent probe length, the position of any matching tags will be found by both threads).

An example of deletion behavior can be seen in the test file ```tests/delete_tests.cu```. To include the TCF with deletes, use the following template:

```
using backing_table = poggers::tables::bucketed_table<
    uint64_t, uint16_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<16, 16>::representation, uint16_t>::representation>::representation,
    4, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 400, poggers::probing_schemes::linearProber,
    poggers::hashers::murmurHasher>;



using del_TCF = poggers::tables::bucketed_table<
    uint64_t, uint16_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<16, 16>::representation, uint16_t>::representation>::representation,
    4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher,
    poggers::hashers::murmurHasher, true, backing_table>;

```

In addition, insertions do not check for tombstones by default. To enable this behavior, use ```tcf->insert_with_delete``` instead of ```tcf->insert```. This comes with a small performance hit of 1.1e9 inserts per second instead of 1.3e9.



FUTURE WORK
________________

2) Dynamic GPU Allocation - In Progress!
3) Host-Device pinned device-side communication
4) Sparse ML components.





PUBLICATIONS