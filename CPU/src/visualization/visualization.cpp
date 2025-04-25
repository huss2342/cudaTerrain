#include "../../include/visualization/visualization.h"
#include "../../include/terrain/terrain_types.h"
#include <stdio.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

void visualizeTerrainChunk(int* terrain, unsigned char* image, int width, int height, int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int terrainType = terrain[idx];
            
            // Get the color from the terrain type
            const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
            
            // Set RGB values in image
            image[idx * 3 + 0] = terrainInfo->color.r;
            image[idx * 3 + 1] = terrainInfo->color.g;
            image[idx * 3 + 2] = terrainInfo->color.b;
        }
    }
}

void visualizeTerrain(int* terrain, unsigned char* image, int width, int height) {
    std::cout << "Starting terrain visualization..." << std::endl;
    
    try {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t idx = (size_t)y * width + x;
                int terrainType = terrain[idx];
                
                // Get the color from the terrain type
                const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
                
                // Set RGB values in image
                image[idx * 3 + 0] = terrainInfo->color.r;
                image[idx * 3 + 1] = terrainInfo->color.g;
                image[idx * 3 + 2] = terrainInfo->color.b;
                
                if ((x == 0 && y == 0) || (x == width-1 && y == height-1)) {
                    std::cout << "Debug - Visualizing position(" << x << "," << y << "): "
                            << "type=" << terrainType 
                            << ", color=(" << terrainInfo->color.r << ","
                            << terrainInfo->color.g << ","
                            << terrainInfo->color.b << ")" << std::endl;
                }
            }
        }
        std::cout << "Visualization completed successfully." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during visualization: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception during visualization!" << std::endl;
    }
}

void visualizeTerrainWithHeightChunk(int* terrain, float* heightMap, unsigned char* image,
                                   int width, int height, int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int terrainType = terrain[idx];
            float height = heightMap[idx];
            
            // Get the base color from the terrain type
            const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
            
            // Adjust color based on height
            float heightFactor = std::max(0.5f, std::min(1.5f, 1.0f + (height - 0.5f)));
            
            // Set RGB values in image with height influence
            image[idx * 3 + 0] = (unsigned char)(terrainInfo->color.r * heightFactor);
            image[idx * 3 + 1] = (unsigned char)(terrainInfo->color.g * heightFactor);
            image[idx * 3 + 2] = (unsigned char)(terrainInfo->color.b * heightFactor);
        }
    }
}

void visualizeTerrainWithHeight(int* terrain, float* heightMap, unsigned char* image, int width, int height) {
    // For now, just use the basic visualization
    visualizeTerrain(terrain, image, width, height);
}

void saveToPPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(image, sizeof(unsigned char), width * height * 3, fp);
    
    fclose(fp);
}