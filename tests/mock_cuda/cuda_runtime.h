#pragma once
#include <cstddef>
#include <cstdint>
typedef int cudaError_t; enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**, size_t); cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemset(void*, int, size_t); cudaError_t cudaDeviceSynchronize();
enum cudaDataType { CUDA_R_64F = 1 };
