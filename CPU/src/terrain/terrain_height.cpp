#include "../../include/terrain/terrain_height.h"
#include "../../include/terrain/terrain_types.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/noise_utils.h"
#include <math.h>
#include <cstring> 

// Helper function to get base height for terrain types - identical to GPU version
float getTerrainBaseHeight(int terrainType) {
    switch(terrainType) {
        // Water bodies - lowest
        case WATER:
        case BAY:
        case FJORD:
        case COVE:
            return 0.0f;
            
        // Beach & coastal
        case BEACH:
        case SAND:
            return 0.05f;
            
        // Plains
        case GRASS:
        case PRAIRIE:
        case SAVANNA:
        case STEPPE:
            return 0.3f;
            
        // Forest & jungle
        case FOREST:
        case JUNGLE:
        case TAIGA:
            return 0.4f;
            
        // Hills & plateaus
        case PLATEAU:
            return 0.6f;
            
        // Mountains
        case MOUNTAIN:
        case ROCK:
        case CLIFF:
            return 0.85f;
            
        // Peaks
        case SNOW:
        case GLACIER:
            return 1.0f;
            
        // Default for other terrains
        default:
            return 0.5f;
    }
}

float blendHeight(float baseHeight, float noiseValue) {
    // Blend the base height (from terrain type) with noise
    // The 0.7/0.3 ratio gives more weight to the terrain type
    return baseHeight * 0.7f + noiseValue * 0.3f;
}

void generateHeightMap(int* terrain, float* heightMap, int width, int height, 
                       float scale, float offsetX, float offsetY) {
    // Process each point in the grid sequentially - replace GPU parallel execution
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Match GPU bounds checking
            if (x < width && y < height) {
                int idx = y * width + x;
                int terrainType = terrain[idx];
                
                // Get base height for this terrain type
                float baseHeight = getTerrainBaseHeight(terrainType);
                
                // Add detailed variation with multiple noise octaves
                float nx = (float)x / width * scale + offsetX;
                float ny = (float)y / height * scale + offsetY;
                
                // Use higher frequency noise for height details
                float detailNoise = distributedNoise(nx * 2.5f, ny * 2.5f, 0.5f, 6);
                
                // Blend base height with noise
                heightMap[idx] = blendHeight(baseHeight, detailNoise);
            }
        }
    }
}

void simulateErosion(float* heightMap, float* output, int width, int height,
                     int iterations, float erosionRate) {
    // Process each point in the grid sequentially - replace GPU parallel execution
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Match GPU bounds checking
            if (x < width && y < height) {
                int idx = y * width + x;
                
                // Skip edges of the map
                if (x <= 2 || y <= 2 || x >= width-3 || y >= height-3) {
                    output[idx] = heightMap[idx];
                    continue;
                }
                
                float currentHeight = heightMap[idx];
                
                // Find steepest downhill direction
                float maxGradient = 0.0f;
                int steepestX = x;
                int steepestY = y;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        int nidx = ny * width + nx;
                        
                        float neighborHeight = heightMap[nidx];
                        float gradient = currentHeight - neighborHeight;
                        
                        if (gradient > maxGradient) {
                            maxGradient = gradient;
                            steepestX = nx;
                            steepestY = ny;
                        }
                    }
                }
                
                // If there's a downhill path, simulate erosion
                if (maxGradient > 0.01f) {
                    // Move some material downhill
                    float material = currentHeight * erosionRate * maxGradient;
                    output[idx] = currentHeight - material;
                    
                    // Deposit some material at the bottom
                    // CPU doesn't need atomicAdd, just add directly
                    int depositIdx = steepestY * width + steepestX;
                    output[depositIdx] += material * 0.8f; // 20% material lost in transit
                } else {
                    output[idx] = currentHeight;
                }
            }
        }
    }
}