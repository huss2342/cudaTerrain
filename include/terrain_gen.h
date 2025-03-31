#ifndef TERRAIN_GEN_H
#define TERRAIN_GEN_H

#include <cuda_runtime.h>

// Kernel function for terrain generation
__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY);

// Helper function to launch the kernel
void createPerlinNoiseTerrain(int* d_terrain, int width, int height, 
                             float scale = 8.0f, float offsetX = 0.0f, 
                             float offsetY = 0.0f);
#endif // TERRAIN_GEN_H