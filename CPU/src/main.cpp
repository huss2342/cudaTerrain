#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "../include/terrain/terrain_types.h"
#include "../include/noise/perlin_noise.h"
#include "../include/terrain/terrain_generator.h"
#include "../include/visualization/visualization.h"
#include "../include/terrain/terrain_height.h"

int main(int argc, char** argv) {
    // Add timing variables at the beginning
    clock_t start_time, end_time;
    double cpu_time_used;
    
    // Start the timer
    start_time = clock();
    
    TerrainTypes::initializeTerrainTypes();
    float scale = 80.0f;  // Default value arg1 - set to 80.0f to match GPU version
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
    
    // Set a specific seed for reproducibility with the GPU version
    int seed = 1234567; // Fixed seed to match GPU version
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
    int floatMemSize = width * height * sizeof(float); // For height map
    
    // Allocate host memory
    int* terrain = new int[width * height];
    unsigned char* image = new unsigned char[width * height * 3];
    float* heightMap = nullptr;
    float* tempHeightMap = nullptr;
    
    if (useHeight) {
        heightMap = new float[width * height];
        tempHeightMap = new float[width * height];
    }
    
    // Generate terrain
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
    delete[] terrain;
    delete[] image;
    if (heightMap) delete[] heightMap;
    if (tempHeightMap) delete[] tempHeightMap;
    
    // Calculate and print execution time
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Total execution time: %.2f seconds\n", cpu_time_used);
    
    return 0;
}