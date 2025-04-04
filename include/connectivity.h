#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include <cuda_runtime.h>

// Function to identify and connect disconnected land areas
void connectLandmasses(int* d_terrain, int width, int height);

__global__ void createVerticalPath(int* d_terrain, int* d_regions, int width, int height, int x);
__global__ void createHorizontalPath(int* d_terrain, int* d_regions, int width, int height, int y);

#endif // CONNECTIVITY_H