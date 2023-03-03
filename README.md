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






FUTURE WORK
________________

2) Dynamic GPU Allocation - In Progress!
3) Host-Device pinned device-side communication
4) Sparse ML components.





PUBLICATIONS