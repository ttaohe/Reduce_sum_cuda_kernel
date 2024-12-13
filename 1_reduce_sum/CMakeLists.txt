
add_executable(reduce_baseline reduce_baseline.cu)
target_link_libraries(reduce_baseline PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_baseline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v0_shared_memory reduce_v0_shared_memory.cu)
target_link_libraries(reduce_v0_shared_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v0_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v1_and reduce_v1_and.cu)
target_link_libraries(reduce_v1_and PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v1_and PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v1_no_warp_divergence reduce_v1_no_warp_divergence.cu)
target_link_libraries(reduce_v1_no_warp_divergence PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v1_no_warp_divergence PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v2_bank_conflict reduce_v2_bank_conflict.cu)
target_link_libraries(reduce_v2_bank_conflict PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v2_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v3_idle reduce_v3_idle.cu)
target_link_libraries(reduce_v3_idle PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v3_idle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v4_last_warp_no_syncthreads reduce_v4_last_warp_no_syncthreads.cu)
target_link_libraries(reduce_v4_last_warp_no_syncthreads PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v4_last_warp_no_syncthreads PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v5_all_expand reduce_v5_all_expand.cu)
target_link_libraries(reduce_v5_all_expand PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v5_all_expand PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v6_blocksize_set reduce_v6_blocksize_set.cu)
target_link_libraries(reduce_v6_blocksize_set PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v6_blocksize_set PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()