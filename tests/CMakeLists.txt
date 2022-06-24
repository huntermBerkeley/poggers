cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



ConfigureExecutable(test_components "${CMAKE_CURRENT_SOURCE_DIR}/src/test_components.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_key_val_pair "${CMAKE_CURRENT_SOURCE_DIR}/src/test_key_val_pair.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(kmer_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/kmer_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(power_2_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/power_2_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

#ConfigureExecutable(separable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/separable_test.cpp" "${POGGERS_TESTS_BINARY_DIR}")

#set_target_properties(separable_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)