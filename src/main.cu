#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "../include/terrain/terrain_types.h"
#include "../include/noise/perlin_noise.h"
#include "../include/terrain/terrain_generator.h"
#include "../include/visualization/visualization.h"

int main(int argc, char** argv) {
    // Parse command line arguments for scale and size
    float scale = 1.0f;  // Default value
    int size = 4096;   // Default size
    
    if (argc > 1) {
        // Try to parse the first argument as the scale
        float inputScale = atof(argv[1]);
        if (inputScale > 0.0f) {
            scale = inputScale;
            printf("Using provided scale: %f\n", scale);
        } else {
            printf("Invalid scale value provided. Using default scale: %f\n", scale);
        }
    } else {
        printf("No scale provided. Using default scale: %f\n", scale);
    }
    
    // Check for size parameter
    if (argc > 2) {
        int inputSize = atoi(argv[2]);
        if (inputSize > 0 && inputSize <= 32000) { // Limit max size to prevent excessive memory usage
            size = inputSize;
            printf("Using provided size: %d x %d\n", size, size);
        } else {
            printf("Invalid or too large size provided. Using default size: %d x %d\n", size, size);
        }
    } else {
        printf("No size provided. Using default size: %d x %d\n", size, size);
    }
    
    // Generate a random seed then split it to X and Y offsets
    int seed = time(NULL); // Random seed based on current time
    srand(seed);
    
    float randomOffsetX = (seed % 100) * 1.27f;
    float randomOffsetY = (seed % 100) * 2.53f; 

    printf("Generated terrain with seed: %d\n", seed);
    printf("Offsets: X=%f, Y=%f\n", randomOffsetX, randomOffsetY);

    // Initialize terrain types
    TerrainTypes::initializeTerrainTypes();

    // Define terrain size
    int width = size;
    int height = size;
    int memSize = width * height * sizeof(int);
    int imageSize = width * height * 3 * sizeof(unsigned char); // RGB
    
    // Allocate host memory
    int* h_terrain = (int*)malloc(memSize);
    unsigned char* h_image = (unsigned char*)malloc(imageSize);
    
    // Allocate device memory
    int* d_terrain;
    unsigned char* d_image;
    cudaMalloc(&d_terrain, memSize);
    cudaMalloc(&d_image, imageSize);
    
    // Generate terrain
    createPerlinNoiseTerrain(d_terrain, width, height, scale, randomOffsetX, randomOffsetY);
   
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // Handle error appropriately
    }

    // Visualize terrain
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    visualizeTerrain<<<gridSize, blockSize>>>(d_terrain, d_image, width, height);
    
    // Copy results back to host
    cudaMemcpy(h_terrain, d_terrain, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    
    // Create output filename with scale and size information
    char filename[100];
    sprintf(filename, "terrain_scale%.1f_size%d.ppm", scale, size);
    
    // Save image to file
    saveToPPM(filename, h_image, width, height);
    
    // Print a message about the improvements with corrected scale explanation
    printf("Generated multi-scale terrain with enhanced detail.\n");
    printf("Scale: %.1f - Lower values = zoomed out (larger features), Higher values = zoomed in (smaller features)\n", scale);
    printf("Saved terrain to %s\n", filename);
    
    // Clean up
    free(h_terrain);
    free(h_image);
    cudaFree(d_terrain);
    cudaFree(d_image);
    
    return 0;
}