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

// Helper function to get base height for terrain types
float getTerrainBaseHeight(int terrainType) {
    switch (terrainType) {
        case 0:  return 0.0f;    // WATER
        case 1:  return 0.1f;    // SAND
        case 2:  return 0.3f;    // GRASS
        case 3:  return 0.5f;    // ROCK
        case 4:  return 0.7f;    // SNOW
        case 5:  return 0.1f;    // LAVA
        case 6:  return 0.2f;    // ICE
        case 7:  return 0.15f;   // MUD
        case 8:  return 0.4f;    // FOREST
        case 9:  return 0.2f;    // DESERT
        case 10: return 0.8f;    // MOUNTAIN
        case 11: return 0.1f;    // SWAMP
        case 12: return 0.45f;   // JUNGLE
        case 13: return 0.25f;   // TUNDRA
        case 14: return 0.35f;   // SAVANNA
        case 15: return 0.4f;    // TAIGA
        case 16: return 0.3f;    // STEPPE
        case 17: return 0.25f;   // PRAIRIE
        case 18: return 0.6f;    // PLATEAU
        case 19: return 0.7f;    // CANYON
        case 20: return 0.5f;    // BADLANDS
        case 21: return 0.55f;   // MESA
        case 22: return 0.15f;   // OASIS
        case 23: return 0.9f;    // VOLCANO
        case 24: return 0.75f;   // GLACIER
        case 25: return 0.4f;    // FJORD
        case 26: return 0.05f;   // BAY
        case 27: return 0.05f;   // COVE
        case 28: return 0.1f;    // BEACH
        case 29: return 0.65f;   // CLIFF
        case 30: return 0.2f;    // DUNE
        default: return 0.3f;    // Default height
    }
}

float blendHeight(float baseHeight, float noiseValue) {
    // Add variation while preserving the base characteristics
    return baseHeight + (noiseValue * 0.3f - 0.15f);
}

// Helper function for multi-threaded height map generation
void generateHeightMapChunk(int* terrain, float* heightMap, int width, int height,
                          float scale, float offsetX, float offsetY,
                          int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
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

void generateHeightMap(int* terrain, float* heightMap, int width, int height,
                      float scale, float offsetX, float offsetY) {
    // Calculate optimal number of threads
    unsigned int maxThreads = std::thread::hardware_concurrency();
    if (maxThreads == 0) maxThreads = 4;
    
    // Limit threads based on image height
    unsigned int numThreads = std::min(maxThreads, (unsigned int)(height / 2));
    numThreads = std::max(1u, numThreads); // Ensure at least 1 thread
    
    std::cout << "Height map generation using " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    for (unsigned int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        
        threads.push_back(std::thread(
            generateHeightMapChunk, terrain, heightMap, width, height,
            scale, offsetX, offsetY, startY, endY
        ));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

// Helper function for multi-threaded erosion simulation
void simulateErosionChunk(float* heightMap, float* output, int width, int height,
                         int iterations, float erosionRate,
                         int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
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
                int depositIdx = steepestY * width + steepestX;
                output[depositIdx] += material * 0.8f; // 20% material lost in transit
            } else {
                output[idx] = currentHeight;
            }
        }
    }
}

void simulateErosion(float* heightMap, float* output, int width, int height,
                    int iterations, float erosionRate) {
    // Calculate optimal number of threads
    unsigned int maxThreads = std::thread::hardware_concurrency();
    if (maxThreads == 0) maxThreads = 4;
    
    // Limit threads based on image height
    unsigned int numThreads = std::min(maxThreads, (unsigned int)(height / 2));
    numThreads = std::max(1u, numThreads); // Ensure at least 1 thread
    
    std::cout << "Erosion simulation using " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    for (unsigned int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        
        threads.push_back(std::thread(
            simulateErosionChunk, heightMap, output, width, height,
            iterations, erosionRate, startY, endY
        ));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}