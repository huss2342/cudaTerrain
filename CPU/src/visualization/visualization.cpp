#include "../../include/visualization/visualization.h"
#include "../../include/terrain/terrain_types.h"
#include <stdio.h>
#include <algorithm> // For min function

void visualizeTerrain(int* terrain, unsigned char* image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < width && y < height) {
                int idx = y * width + x;
                int terrainType = terrain[idx];
                
                // Get the color from the terrain type
                const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
                
                // Set RGB values in image (assuming 3 channels)
                image[idx * 3 + 0] = terrainInfo->color.r;
                image[idx * 3 + 1] = terrainInfo->color.g;
                image[idx * 3 + 2] = terrainInfo->color.b;
            }
        }
    }
}


void visualizeTerrainWithHeight(int* terrain, float* heightMap, unsigned char* image, int width, int height) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            if (x < width && y < height) {
                int idx = y * width + x;
                int terrainType = terrain[idx];
                float elevation = heightMap[idx];
                
                // Get the base color from the terrain type
                const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
                
                // Calculate simple shading based on height gradient
                // Calculate directional shading (from northwest light source)
                float shadingIntensity = 1.0f; // Default intensity
                
                // Check if we can compute gradient (not on edges)
                if (x > 0 && y > 0 && x < width-1 && y < height-1) {
                    // Get heights of neighboring pixels
                    float heightN = heightMap[(y-1) * width + x];
                    float heightW = heightMap[y * width + (x-1)];
                    
                    // Calculate slope (simplified gradient)
                    float gradientX = elevation - heightW;
                    float gradientY = elevation - heightN;
                    
                    // Light direction (normalized vector pointing southeast)
                    float lightX = 0.7071f; // 1/sqrt(2)
                    float lightY = 0.7071f;
                    
                    // Dot product between gradient and light
                    float dotProduct = (gradientX * lightX + gradientY * lightY);
                    
                    // Convert to shading intensity (range 0.6 to 1.2)
                    shadingIntensity = 0.9f + dotProduct * 2.0f;
                    if (shadingIntensity < 0.6f) shadingIntensity = 0.6f;
                    if (shadingIntensity > 1.2f) shadingIntensity = 1.2f;
                }
                
                // Apply shading to RGB values
                int r = std::min(255, (int)(terrainInfo->color.r * shadingIntensity));
                int g = std::min(255, (int)(terrainInfo->color.g * shadingIntensity));
                int b = std::min(255, (int)(terrainInfo->color.b * shadingIntensity));
                
                // Set RGB values in image
                image[idx * 3 + 0] = r;
                image[idx * 3 + 1] = g;
                image[idx * 3 + 2] = b;
            }
        }
    }
}

void saveToPPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(image, 3, width * height, fp);
    
    fclose(fp);
    printf("Saved terrain image to %s\n", filename);
}