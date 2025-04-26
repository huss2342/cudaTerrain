#include "../../include/terrain/terrain_smoothing.h"
#include <math.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <stdexcept>
#include <iostream>

// Multi-threaded smoothing helper function
void smoothTerrainChunk(int* terrain, int* output, int width, int height, int startY, int endY) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in smoothTerrainChunk");
    }

    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            long long idx = (long long)y * width + x;
            if (idx < 0 || idx >= (long long)width * height) {
                throw std::runtime_error("Index out of bounds in smoothTerrainChunk");
            }
            
            // Skip edges
            if (x <= 1 || y <= 1 || x >= width-2 || y >= height-2) {
                output[idx] = terrain[idx];
                continue;
            }
            
            // Count occurrences of neighboring terrain types
            int typeCounts[31] = {0};
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int nx = std::max(0, std::min(width-1, x + dx));
                    int ny = y + dy;
                    long long nidx = (long long)ny * width + nx;
                    
                    if (nidx >= 0 && nidx < (long long)width * height) {
                        typeCounts[terrain[nidx]]++;
                    }
                }
            }
            
            // Find most common type
            int bestType = terrain[idx];
            int maxCount = 0;
            
            for (int t = 0; t < 31; t++) {
                if (typeCounts[t] > maxCount) {
                    maxCount = typeCounts[t];
                    bestType = t;
                }
            }
            
            output[idx] = bestType;
        }
    }
}

void smoothTerrain(int* terrain, int* output, int width, int height) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in smoothTerrain");
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in smoothTerrain");
    }

    // For small terrains, just process single-threaded
    if (width * height <= 1024) {
        smoothTerrainChunk(terrain, output, width, height, 0, height);
        return;
    }

    // Use multithreading for larger terrains
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    // Ensure each thread has enough work
    numThreads = std::min(numThreads, (unsigned int)(height / 32));
    if (numThreads == 0) numThreads = 1;

    std::cout << "Smoothing with " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    try {
        for (unsigned int i = 0; i < numThreads; i++) {
            int startY = i * rowsPerThread;
            int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
            
            threads.push_back(std::thread(
                smoothTerrainChunk, terrain, output, width, height, startY, endY
            ));
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in smoothTerrain: " << e.what() << std::endl;
        throw;
    }
}

void removeVerticalStripesChunk(int* terrain, int* output, int width, int height, int startY, int endY) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in removeVerticalStripesChunk");
    }

    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            long long idx = (long long)y * width + x;
            if (idx < 0 || idx >= (long long)width * height) {
                throw std::runtime_error("Index out of bounds in removeVerticalStripesChunk");
            }
            
            // Skip edges
            if (x <= 2 || x >= width-3) {
                output[idx] = terrain[idx];
                continue;
            }
            
            // Look at horizontal neighbors only
            int currentType = terrain[idx];
            
            // Safely get neighbor indices
            long long leftIdx1 = idx - 1;
            long long leftIdx2 = idx - 2;
            long long rightIdx1 = idx + 1;
            long long rightIdx2 = idx + 2;
            
            if (leftIdx1 < 0 || leftIdx2 < 0 || rightIdx1 >= (long long)width * height || rightIdx2 >= (long long)width * height) {
                output[idx] = currentType;
                continue;
            }
            
            int leftType1 = terrain[leftIdx1];
            int leftType2 = terrain[leftIdx2];
            int rightType1 = terrain[rightIdx1];
            int rightType2 = terrain[rightIdx2];
            
            if (leftType1 == leftType2 && rightType1 == rightType2 && 
                leftType1 != currentType && rightType1 != currentType) {
                int counts[31] = {0};
                counts[leftType1] += 2;
                counts[rightType1] += 2;
                
                int bestType = currentType;
                int maxCount = 0;
                for (int t = 0; t < 31; t++) {
                    if (counts[t] > maxCount) {
                        maxCount = counts[t];
                        bestType = t;
                    }
                }
                output[idx] = bestType;
            } else {
                output[idx] = currentType;
            }
        }
    }
}

void removeVerticalStripes(int* terrain, int* output, int width, int height) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in removeVerticalStripes");
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in removeVerticalStripes");
    }

    // For small terrains, just process single-threaded
    if (width * height <= 1024) {
        removeVerticalStripesChunk(terrain, output, width, height, 0, height);
        return;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    // Ensure each thread has enough work
    numThreads = std::min(numThreads, (unsigned int)(height / 32));
    if (numThreads == 0) numThreads = 1;

    std::cout << "Removing vertical stripes with " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    try {
        for (unsigned int i = 0; i < numThreads; i++) {
            int startY = i * rowsPerThread;
            int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
            
            threads.push_back(std::thread(
                removeVerticalStripesChunk, terrain, output, width, height, startY, endY
            ));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in removeVerticalStripes: " << e.what() << std::endl;
        throw;
    }
}

void removeIsolatedNoiseChunk(int* terrain, int* output, int width, int height, int startY, int endY) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in removeIsolatedNoiseChunk");
    }

    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            long long idx = (long long)y * width + x;
            if (idx < 0 || idx >= (long long)width * height) {
                throw std::runtime_error("Index out of bounds in removeIsolatedNoiseChunk");
            }
            
            // Skip edges
            if (x <= 1 || y <= 1 || x >= width-2 || y >= height-2) {
                output[idx] = terrain[idx];
                continue;
            }
            
            int currentType = terrain[idx];
            int sameTypeCount = 0;
            
            // Check immediate neighbors only (3x3 area)
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    
                    long long nidx = (long long)ny * width + nx;
                    if (nidx >= 0 && nidx < (long long)width * height) {
                        if (terrain[nidx] == currentType) {
                            sameTypeCount++;
                        }
                    }
                }
            }
            
            if (sameTypeCount <= 2) {
                int typeCounts[31] = {0};
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        
                        long long nidx = (long long)ny * width + nx;
                        if (nidx >= 0 && nidx < (long long)width * height) {
                            if (terrain[nidx] != currentType) {
                                typeCounts[terrain[nidx]]++;
                            }
                        }
                    }
                }
                
                int bestType = currentType;
                int maxCount = 0;
                
                for (int t = 0; t < 31; t++) {
                    if (t != currentType && typeCounts[t] > maxCount) {
                        maxCount = typeCounts[t];
                        bestType = t;
                    }
                }
                
                output[idx] = bestType;
            } else {
                output[idx] = currentType;
            }
        }
    }
}

void removeIsolatedNoise(int* terrain, int* output, int width, int height) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in removeIsolatedNoise");
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in removeIsolatedNoise");
    }

    // For small terrains, just process single-threaded
    if (width * height <= 1024) {
        removeIsolatedNoiseChunk(terrain, output, width, height, 0, height);
        return;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    // Ensure each thread has enough work
    numThreads = std::min(numThreads, (unsigned int)(height / 32));
    if (numThreads == 0) numThreads = 1;

    std::cout << "Removing isolated noise with " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    try {
        for (unsigned int i = 0; i < numThreads; i++) {
            int startY = i * rowsPerThread;
            int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
            
            threads.push_back(std::thread(
                removeIsolatedNoiseChunk, terrain, output, width, height, startY, endY
            ));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in removeIsolatedNoise: " << e.what() << std::endl;
        throw;
    }
}

void improvedSmoothTerrainChunk(int* terrain, int* output, int width, int height, int startY, int endY) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in improvedSmoothTerrainChunk");
    }

    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            long long idx = (long long)y * width + x;
            if (idx < 0 || idx >= (long long)width * height) {
                throw std::runtime_error("Index out of bounds in improvedSmoothTerrainChunk");
            }
            
            // Skip edges
            if (x <= 1 || y <= 1 || x >= width-2 || y >= height-2) {
                output[idx] = terrain[idx];
                continue;
            }
            
            int currentType = terrain[idx];
            int typeCounts[31] = {0};
            
            // Use a smaller radius for better performance and safety
            const int radius = std::min(10, std::min(width/4, height/4));
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    
                    long long nidx = (long long)ny * width + nx;
                    if (nidx < 0 || nidx >= (long long)width * height) {
                        continue;
                    }
                    
                    int neighborType = terrain[nidx];
                    float dist = sqrtf((float)(dx*dx + dy*dy));
                    int weight = 1;
                    
                    if (neighborType == currentType) {
                        weight = 3 + (radius - (int)dist);
                    } else {
                        weight = radius - (int)dist;
                        if (weight <= 0) weight = 1;
                    }
                    
                    typeCounts[neighborType] += weight;
                }
            }
            
            int bestType = currentType;
            int maxCount = 0;
            
            for (int t = 0; t < 31; t++) {
                if (typeCounts[t] > maxCount) {
                    maxCount = typeCounts[t];
                    bestType = t;
                }
            }
            
            output[idx] = bestType;
        }
    }
}

void improvedSmoothTerrain(int* terrain, int* output, int width, int height) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in improvedSmoothTerrain");
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in improvedSmoothTerrain");
    }

    // For small terrains, just process single-threaded
    if (width * height <= 1024) {
        improvedSmoothTerrainChunk(terrain, output, width, height, 0, height);
        return;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    // Ensure each thread has enough work
    numThreads = std::min(numThreads, (unsigned int)(height / 32));
    if (numThreads == 0) numThreads = 1;

    std::cout << "Improved smoothing with " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    try {
        for (unsigned int i = 0; i < numThreads; i++) {
            int startY = i * rowsPerThread;
            int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
            
            threads.push_back(std::thread(
                improvedSmoothTerrainChunk, terrain, output, width, height, startY, endY
            ));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in improvedSmoothTerrain: " << e.what() << std::endl;
        throw;
    }
}

void cleanupSmallPatchesChunk(int* terrain, int* output, int width, int height, int minRegionSize, int startY, int endY) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in cleanupSmallPatchesChunk");
    }

    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            long long idx = (long long)y * width + x;
            if (idx < 0 || idx >= (long long)width * height) {
                throw std::runtime_error("Index out of bounds in cleanupSmallPatchesChunk");
            }
            
            if (x <= 2 || y <= 2 || x >= width-3 || y >= height-3) {
                output[idx] = terrain[idx];
                continue;
            }
            
            int currentType = terrain[idx];
            int sameTypeCount = 0;
            int totalCount = 0;
            int neighborTypes[31] = {0};
            
            // Use a smaller filter size for better performance and safety
            const int filterSize = std::min(20, std::min(width/4, height/4));
            const int halfFilterSize = filterSize / 2;

            for (int dy = -halfFilterSize; dy <= halfFilterSize; dy++) {
                for (int dx = -halfFilterSize; dx <= halfFilterSize; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    
                    long long nidx = (long long)ny * width + nx;
                    if (nidx < 0 || nidx >= (long long)width * height) {
                        continue;
                    }
                    
                    int neighborType = terrain[nidx];
                    totalCount++;
                    neighborTypes[neighborType]++;
                    
                    if (neighborType == currentType) {
                        sameTypeCount++;
                    }
                }
            }
            
            if (totalCount > 0) {
                float sameRatio = (float)sameTypeCount / totalCount;
                if (sameRatio < (float)minRegionSize / 100.0f) {
                    int bestType = currentType;
                    int maxCount = 0;
                    
                    for (int t = 0; t < 31; t++) {
                        if (neighborTypes[t] > maxCount) {
                            maxCount = neighborTypes[t];
                            bestType = t;
                        }
                    }
                    
                    output[idx] = bestType;
                } else {
                    output[idx] = currentType;
                }
            } else {
                output[idx] = currentType;
            }
        }
    }
}

void cleanupSmallPatches(int* terrain, int* output, int width, int height, int minRegionSize) {
    if (!terrain || !output) {
        throw std::runtime_error("Null pointer in cleanupSmallPatches");
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in cleanupSmallPatches");
    }

    // For small terrains, just process single-threaded
    if (width * height <= 1024) {
        cleanupSmallPatchesChunk(terrain, output, width, height, minRegionSize, 0, height);
        return;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    // Ensure each thread has enough work
    numThreads = std::min(numThreads, (unsigned int)(height / 32));
    if (numThreads == 0) numThreads = 1;

    std::cout << "Cleaning up small patches with " << numThreads << " threads" << std::endl;
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    try {
        for (unsigned int i = 0; i < numThreads; i++) {
            int startY = i * rowsPerThread;
            int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
            
            threads.push_back(std::thread(
                cleanupSmallPatchesChunk, terrain, output, width, height, minRegionSize, startY, endY
            ));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in cleanupSmallPatches: " << e.what() << std::endl;
        throw;
    }
}