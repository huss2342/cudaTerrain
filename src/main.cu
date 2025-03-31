#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "../include/terrain_types.h"
#include "../include/perlin_noise.h"
#include "../include/terrain_gen.h"
#include "../include/visualization.h"

int main() {
    // Initialize terrain types
    TerrainTypes::initializeTerrainTypes();

    // Define terrain size
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(int);
    int imageSize = width * height * 3 * sizeof(unsigned char); // RGB
    
    // Allocate host memory
    int* h_terrain = (int*)malloc(size);
    unsigned char* h_image = (unsigned char*)malloc(imageSize);
    
    // Allocate device memory
    int* d_terrain;
    unsigned char* d_image;
    cudaMalloc(&d_terrain, size);
    cudaMalloc(&d_image, imageSize);
    
    // Generate terrain
    createPerlinNoiseTerrain(d_terrain, width, height);
    
    // Visualize terrain
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    visualizeTerrain<<<gridSize, blockSize>>>(d_terrain, d_image, width, height);
    
    // Copy results back to host
    cudaMemcpy(h_terrain, d_terrain, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    
    // Save image to file
    saveToPPM("terrain.ppm", h_image, width, height);
    
    // Clean up
    free(h_terrain);
    free(h_image);
    cudaFree(d_terrain);
    cudaFree(d_image);
    
    return 0;
}