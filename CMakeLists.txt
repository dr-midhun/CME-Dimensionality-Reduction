cmake_minimum_required(VERSION 3.18)
project(Dimensionality_Reduction_of_the_Chemical_Master_Equation LANGUAGES CXX)

option(CME_ENABLE_CUDA   "Build the GPU backend (cuBLAS + cuSPARSE)" OFF)
option(CME_ENABLE_LAPACK "Use LAPACK dgeev for the d x d reduced eigenproblem" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(OpenMP REQUIRED)

add_library(cme STATIC src/network.cpp src/csr_util.cpp
                       src/alg1_cme.cpp
                       src/alg2_state_space.cpp
                       src/alg3_transition.cpp
                       src/alg4_reduction.cpp
                       src/alg5_eigenbasis.cpp src/alg5_eigenbasis_cpu.cpp
                       src/alg6_7_8_fsp.cpp
                       src/alg9_10_11_12_preconditioned_fsp.cpp)
target_include_directories(cme PUBLIC include)
target_link_libraries(cme PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(cme PRIVATE -Wall -Wextra)

if(CME_ENABLE_LAPACK)
  find_package(LAPACK)
  if(LAPACK_FOUND)
    target_compile_definitions(cme PUBLIC CME_WITH_LAPACK)
    target_link_libraries(cme PUBLIC ${LAPACK_LIBRARIES})
  else()
    message(WARNING "LAPACK not found: eigenvalues of the reduced matrix will not be computed")
  endif()
endif()

if(CME_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(cme PRIVATE src/alg5_eigenbasis_cuda.cpp)
  target_compile_definitions(cme PUBLIC CME_WITH_CUDA)
  target_link_libraries(cme PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
endif()

add_executable(cme_cli apps/cme_cli.cpp)
target_link_libraries(cme_cli PRIVATE cme)
add_executable(test_small tests/test_small.cpp)
target_link_libraries(test_small PRIVATE cme)
enable_testing()
add_test(NAME thesis_mm_cpu COMMAND test_small)
if(CME_ENABLE_CUDA)
  add_test(NAME thesis_mm_gpu COMMAND test_small gpu)
endif()
