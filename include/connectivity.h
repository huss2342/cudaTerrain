#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H
#include <cuda_runtime.h>

// Function to identify and connect disconnected land areas
void connectLandmasses(int* d_terrain, int width, int height);

// Add missing declaration
void createConnectivityNetwork(int* d_terrain, int* d_regions, int width, int height, int maxRegion);

// Fix __global__ keyword (double underscores)
__global__ void createVerticalPath(int* d_terrain, int* d_regions, int width, int height, int x);
__global__ void createHorizontalPath(int* d_terrain, int* d_regions, int width, int height, int y);

// Add missing kernel declaration
__global__ void createPathways(int* terrain, int* regions, int width, int height, 
                               int region1, int region2, int pathX1, int pathY1, 
                               int pathX2, int pathY2);

#endif // CONNECTIVITY_H