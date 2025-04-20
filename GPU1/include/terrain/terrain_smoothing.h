#ifndef TERRAIN_SMOOTHING_H
#define TERRAIN_SMOOTHING_H

#include <cuda_runtime.h>

// Smoothing operations
__global__ void smoothTerrain(int* terrain, int* output, int width, int height);
__global__ void improvedSmoothTerrain(int* terrain, int* output, int width, int height);
__global__ void removeVerticalStripes(int* terrain, int* output, int width, int height);
__global__ void removeIsolatedNoise(int* terrain, int* output, int width, int height);
__global__ void cleanupSmallPatches(int* terrain, int* output, int width, int height, int minRegionSize);

#endif // TERRAIN_SMOOTHING_H