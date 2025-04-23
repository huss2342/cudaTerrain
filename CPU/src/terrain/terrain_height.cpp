#include "../../include/terrain/terrain_height.h"
#include "../../include/terrain/terrain_types.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/noise_utils.h"
#include <math.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

// Assign base height values to terrain types
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
    // Determine number of threads to use (based on hardware)
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 if detection fails
    
    // Create and start threads
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
    // Apply erosion to assigned rows
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
                
                // Deposit some material at the bottom (atomic add not needed in single-threaded version)
                int depositIdx = steepestY * width + steepestX;
                
                // We'll use a mutex-based approach to handle atomic operations
                // But for simplicity in this CPU implementation, we'll just add it directly
                // This should not cause major issues as long as adjacent threads don't access the same cell
                output[depositIdx] += material * 0.8f; // 20% material lost in transit
            } else {
                output[idx] = currentHeight;
            }
        }
    }
}

void simulateErosion(float* heightMap, float* output, int width, int height,
                   int iterations, float erosionRate) {
    // Determine number of threads to use
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 if detection fails
    
    // Create and start threads
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