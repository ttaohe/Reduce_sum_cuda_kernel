cmake_minimum_required(VERSION 3.17.0)
project(universe_cuda_best_practice VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)
add_subdirectory(1_reduce_sum)
add_subdirectory(2_sgemm)
