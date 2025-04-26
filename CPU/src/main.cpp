#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cstring>

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
    long long memSize = (long long)width * height * sizeof(int);
    long long imageSize = (long long)width * height * 3 * sizeof(unsigned char); // RGB
    long long floatMemSize = (long long)width * height * sizeof(float); // For height map
    
    // Allocate host memory
    int* terrain = (int*)malloc(memSize);
    unsigned char* image = (unsigned char*)malloc(imageSize);
    float* heightMap = NULL;
    float* tempHeightMap = NULL;
    
    if (useHeight) {
        heightMap = (float*)malloc(floatMemSize);
        tempHeightMap = (float*)malloc(floatMemSize); // Equivalent to d_tempHeightMap
    }
    
    if (!terrain || !image || (useHeight && (!heightMap || !tempHeightMap))) {
        printf("Failed to allocate memory\n");
        // Cleanup any allocated memory
        if (terrain) free(terrain);
        if (image) free(image);
        if (heightMap) free(heightMap);
        if (tempHeightMap) free(tempHeightMap);
        return 1;
    }
    
    // Generate terrain
    printf("Generating terrain...\n");
    createPerlinNoiseTerrain(terrain, width, height, scale, randomOffsetX, randomOffsetY);
    
    if (useHeight) {
        // Generate height map based on terrain types
        printf("Generating height map...\n");
        generateHeightMap(terrain, heightMap, width, height, scale, randomOffsetX, randomOffsetY);
        
        // Apply erosion simulation (multiple passes)
        printf("Simulating erosion...\n");
        const int erosionIterations = 5;
        const float erosionRate = 0.1f;
        for (int i = 0; i < erosionIterations; i++) {
            simulateErosion(heightMap, tempHeightMap, width, height, 1, erosionRate);
            // Copy tempHeightMap back to heightMap (equivalent to cudaMemcpy)
            memcpy(heightMap, tempHeightMap, floatMemSize);
        }
        
        // Visualize terrain with height information
        printf("Visualizing terrain with height information...\n");
        visualizeTerrainWithHeight(terrain, heightMap, image, width, height);
    } else {
        // Use the original visualization without height
        printf("Visualizing terrain without height information...\n");
        visualizeTerrain(terrain, image, width, height);
    }
    
    // Create output filename with scale and size information
    char filename[100];
    sprintf(filename, "terrain_scale%.1f_size%d%s.ppm", scale, size, useHeight ? "_height" : "");
    
    // Save image to file
    saveToPPM(filename, image, width, height);
    
    // Print a message about the improvements with corrected scale explanation
    printf("Generated multi-scale terrain with enhanced detail.\n");
    printf("Scale: %.1f - Lower values = zoomed out (larger features), Higher values = zoomed in (smaller features)\n", scale);
    printf("Saved terrain to %s\n", filename);
    
    // Clean up
    free(terrain);
    free(image);
    if (heightMap) free(heightMap);
    if (tempHeightMap) free(tempHeightMap);
    
    // Calculate and print execution time
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Total execution time: %.2f seconds\n", cpu_time_used);
    
    return 0;
}