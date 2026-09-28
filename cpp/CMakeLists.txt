cmake_minimum_required(VERSION 3.18)
# Phase 2 client. Built inside the triton-bench:v3 image (Triton C++ client SDK,
# CUDA), the same image as the companion repo's clients; see scripts/phase2_build.sh.
project(inspect_client LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
set(TRITON_CLIENT_ROOT "/workspace/install" CACHE PATH "Triton C++ client SDK root")
find_path(TRITON_CLIENT_INC grpc_client.h
  HINTS ${TRITON_CLIENT_ROOT}/include /opt/tritonclient/include /opt/include)
find_library(TRITON_GRPC_LIB grpcclient
  HINTS ${TRITON_CLIENT_ROOT}/lib /opt/tritonclient/lib /opt/lib)
if(NOT TRITON_CLIENT_INC OR NOT TRITON_GRPC_LIB)
  message(FATAL_ERROR "Triton C++ client SDK not found; set -DTRITON_CLIENT_ROOT")
endif()
add_executable(inspect_client inspect_client.cu)
target_include_directories(inspect_client PRIVATE ${TRITON_CLIENT_INC})
target_link_libraries(inspect_client PRIVATE ${TRITON_GRPC_LIB} CUDA::cudart)
