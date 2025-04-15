#ifndef TERRAIN_HEIGHT_H
#define TERRAIN_HEIGHT_H

#include <cuda_runtime.h>

// Height generation and manipulation kernels
__global__ void generateHeightMap(int* terrain, float* heightMap, int width, int height, 
                                  float scale, float offsetX, float offsetY);

// Erosion simulation
__global__ void simulateErosion(float* heightMap, float* output, int width, int height,
                               int iterations, float erosionRate);

// Utility functions for height operations
__device__ float getTerrainBaseHeight(int terrainType);
__device__ float blendHeight(float baseHeight, float noiseValue);

#endif // TERRAIN_HEIGHT_H