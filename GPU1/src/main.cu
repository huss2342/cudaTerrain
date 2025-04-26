#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>

#include "../include/terrain/terrain_types.h"
#include "../include/noise/perlin_noise.h"
#include "../include/terrain/terrain_generator.h"
#include "../include/visualization/visualization.h"
#include "../include/terrain/terrain_height.h"

int main(int argc, char** argv) {
    clock_t start_time, end_time;
    double cpu_time_used;
    
    // Start the timer
    start_time = clock();
    
    TerrainTypes::initializeTerrainTypes();
    float scale = 80.0f;  // Default value arg1
    int size = 4096;     // Default size arg2
    bool useHeight = true; // Default to using height visualization
    
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
        if (inputSize > 0 && inputSize <= 22000) { // Limit max size to prevent excessive memory usage
            size = inputSize;
            printf("Using provided size: %d x %d\n", size, size);
        } else {
            printf("Invalid or too large size provided. Using default size: %d x %d\n", size, size);
        }
    } else {
        printf("No size provided. Using default size: %d x %d\n", size, size);
    }
    
    // Optional parameter to disable height visualization
    if (argc > 3) {
        if (strcmp(argv[3], "noheight") == 0) {
            useHeight = false;
            printf("Height visualization disabled\n");
        }
    }
    
    // Generate a random seed then split it to X and Y offsets
    int seed = time(NULL); // Random seed based on current time
    seed = 1234567; // Override with fixed seed to match CPU&GPU version
    srand(seed);
    
    float randomOffsetX = (seed % 100) * 1.27f;
    float randomOffsetY = (seed % 100) * 2.53f; 

    printf("Generated terrain with seed: %d\n", seed);
    printf("Offsets: X=%f, Y=%f\n", randomOffsetX, randomOffsetY);

    //TerrainTypes::initializeTerrainTypes();

    // Define terrain size
    int width = size;
    int height = size;
    int memSize = width * height * sizeof(int);
    int imageSize = width * height * 3 * sizeof(unsigned char); // RGB
    int floatMemSize = width * height * sizeof(float); // For height map
    
    // Allocate host memory
    int* h_terrain = (int*)malloc(memSize);
    unsigned char* h_image = (unsigned char*)malloc(imageSize);
    float* h_heightMap = NULL;
    
    if (useHeight) {
        h_heightMap = (float*)malloc(floatMemSize);
    }
    
    // Allocate device memory
    int* d_terrain;
    unsigned char* d_image;
    float* d_heightMap = NULL;
    float* d_tempHeightMap = NULL;
    
    cudaMalloc(&d_terrain, memSize);
    cudaMalloc(&d_image, imageSize);
    
    if (useHeight) {
        cudaMalloc(&d_heightMap, floatMemSize);
        cudaMalloc(&d_tempHeightMap, floatMemSize);
    }
    
    // Generate terrain
    createPerlinNoiseTerrain(d_terrain, width, height, scale, randomOffsetX, randomOffsetY);
   
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // Handle error appropriately
    }

    // Set up CUDA execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    if (useHeight) {
        // Generate height map based on terrain types
        printf("Generating height map...\n");
        generateHeightMap<<<gridSize, blockSize>>>(d_terrain, d_heightMap, width, height, scale, randomOffsetX, randomOffsetY);
        
        // Apply erosion simulation (multiple passes)
        printf("Simulating erosion...\n");
        const int erosionIterations = 5;
        const float erosionRate = 0.1f;
        for (int i = 0; i < erosionIterations; i++) {
            simulateErosion<<<gridSize, blockSize>>>(d_heightMap, d_tempHeightMap, width, height, 1, erosionRate);
            cudaMemcpy(d_heightMap, d_tempHeightMap, floatMemSize, cudaMemcpyDeviceToDevice);
        }
        
        // Visualize terrain with height information
        printf("Visualizing terrain with height information...\n");
        visualizeTerrainWithHeight<<<gridSize, blockSize>>>(d_terrain, d_heightMap, d_image, width, height);
        
        // Copy height map back to host (if you need it for additional processing)
        cudaMemcpy(h_heightMap, d_heightMap, floatMemSize, cudaMemcpyDeviceToHost);
    } else {
        // Use the original visualization without height
        printf("Visualizing terrain without height information...\n");
        visualizeTerrain<<<gridSize, blockSize>>>(d_terrain, d_image, width, height);
    }
    
    // Copy terrain and image results back to host
    cudaMemcpy(h_terrain, d_terrain, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    
    // Create output filename with scale and size information
    char filename[100];
    sprintf(filename, "terrain_scale%.1f_size%d%s.ppm", scale, size, useHeight ? "_height" : "");
    
    // Save image to file
    saveToPPM(filename, h_image, width, height);
    
    // Print a message about the improvements with corrected scale explanation
    printf("Generated multi-scale terrain with enhanced detail.\n");
    printf("Scale: %.1f - Lower values = zoomed out (larger features), Higher values = zoomed in (smaller features)\n", scale);
    printf("Saved terrain to %s\n", filename);
    
    // Clean up
    free(h_terrain);
    free(h_image);
    if (h_heightMap) free(h_heightMap);
    
    cudaFree(d_terrain);
    cudaFree(d_image);
    if (d_heightMap) cudaFree(d_heightMap);
    if (d_tempHeightMap) cudaFree(d_tempHeightMap);
    
    // Calculate and print execution time
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Total execution time: %.2f seconds\n", cpu_time_used);
    
    return 0;
}