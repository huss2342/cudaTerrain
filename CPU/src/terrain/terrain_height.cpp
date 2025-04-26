#include "../../include/terrain/terrain_height.h"
#include "../../include/terrain/terrain_types.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/noise_utils.h"
#include <math.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <stdexcept>

// Helper function to get base height for terrain types
float getTerrainBaseHeight(int terrainType) {
    switch (terrainType) {
        case WATER:
        case BAY:
        case FJORD:
        case COVE:
            return 0.0f;
        case BEACH:
        case SAND:
            return 0.05f;
        case GRASS:
        case PRAIRIE:
        case SAVANNA:
        case STEPPE:
            return 0.3f;
        case FOREST:
        case JUNGLE:
        case TAIGA:
            return 0.4f;
        case PLATEAU:
            return 0.6f;
        case MOUNTAIN:
        case ROCK:
        case CLIFF:
            return 0.85f;
        case SNOW:
        case GLACIER:
            return 1.0f;
        default:
            return 0.5f;
    }
}

float blendHeight(float baseHeight, float noiseValue) {
    // Blend the base height with noise using same ratio as GPU
    return baseHeight * 0.7f + noiseValue * 0.3f;
}

void generateHeightMap(int* terrain, float* heightMap, int width, int height, 
                      float scale, float offsetX, float offsetY) {
    if (!terrain || !heightMap) {
        throw std::runtime_error("Null pointer passed to generateHeightMap");
    }
    
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in generateHeightMap");
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            long long idx = (long long)y * width + x;
            if (idx >= (long long)width * height) {
                throw std::runtime_error("Index out of bounds in generateHeightMap");
            }
            
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

void simulateErosion(float* heightMap, float* output, int width, int height,
                     int iterations, float erosionRate) {
    if (!heightMap || !output) {
        throw std::runtime_error("Null pointer passed to simulateErosion");
    }
    
    if (width <= 4 || height <= 4) {
        throw std::runtime_error("Dimensions too small for erosion simulation");
    }
    
    // Copy initial heightmap to output
    std::copy(heightMap, heightMap + (long long)width * height, output);
    
    // Clamp erosion rate to prevent instability
    float clampedErosionRate = std::min(1.0f, std::max(0.0f, erosionRate));
    
    for (int iter = 0; iter < iterations; iter++) {
        // Use a 2-pixel border to prevent edge issues
        for (int y = 2; y < height-2; y++) {
            for (int x = 2; x < width-2; x++) {
                long long idx = (long long)y * width + x;
                if (idx >= (long long)width * height) {
                    throw std::runtime_error("Index out of bounds in simulateErosion");
                }
                
                float currentHeight = output[idx];
                float maxGradient = 0.0f;
                int steepestX = x;
                int steepestY = y;
                
                // Find steepest downhill direction
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        long long nidx = (long long)ny * width + nx;
                        
                        if (nidx < 0 || nidx >= (long long)width * height) {
                            continue;
                        }
                        
                        float neighborHeight = output[nidx];
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
                    float material = currentHeight * clampedErosionRate * maxGradient;
                    output[idx] = currentHeight - material;
                    
                    long long depositIdx = (long long)steepestY * width + steepestX;
                    if (depositIdx >= 0 && depositIdx < (long long)width * height) {
                        output[depositIdx] += material * 0.8f;
                    }
                }
            }
        }
    }
}