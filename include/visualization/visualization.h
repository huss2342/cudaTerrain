#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <cuda_runtime.h>

// Kernel for terrain visualization
__global__ void visualizeTerrain(int* terrain, unsigned char* image, int width, int height);

// Updated kernel that includes height visualization
__global__ void visualizeTerrainWithHeight(int* terrain, float* heightMap, unsigned char* image, int width, int height);

// Function to save image to a PPM file
void saveToPPM(const char* filename, unsigned char* image, int width, int height);

#endif // VISUALIZATION_H