#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#include <cuda_runtime.h>

// Perlin noise helper functions
__device__ float fade(float t);
__device__ float lerp(float a, float b, float t);
__device__ float grad(int hash, float x, float y, float z);
__device__ float noise(float x, float y, float z);

#endif // PERLIN_NOISE_H