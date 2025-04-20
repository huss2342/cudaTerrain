#ifndef NOISE_UTILS_H
#define NOISE_UTILS_H

#include <cuda_runtime.h>

// Enhanced noise functions
__device__ float enhancedNoise(float x, float y, float z);
__device__ float distributedNoise(float x, float y, float z, int octaves);

#endif // NOISE_UTILS_H