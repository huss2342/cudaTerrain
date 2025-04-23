// CPU/src/terrain/component_analysis.cpp
#include "../../include/terrain/component_analysis.h"
#include <thread>
#include <vector>
#include <cstring>

void identifyConnectedComponentsChunk(int* terrain, int* labels, int width, int height, int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int currentType = terrain[idx];
            
            // Initial label is our unique index
            labels[idx] = idx;
            
            // Look at 4-connected neighbors that have already been processed
            // This is just the north and west neighbors due to the processing order
            if (y > 0) {
                int northIdx = (y-1) * width + x;
                if (terrain[northIdx] == currentType) {
                    // Link to north's label
                    labels[idx] = labels[northIdx];
                }
            }
            
            if (x > 0) {
                int westIdx = y * width + (x-1);
                if (terrain[westIdx] == currentType) {
                    // Link to west's label, or if already linked to north,
                    // take the smaller of the two labels
                    if (labels[idx] == idx || labels[westIdx] < labels[idx]) {
                        labels[idx] = labels[westIdx];
                    }
                }
            }
        }
    }
}

void identifyConnectedComponents(int* terrain, int* labels, int width, int height) {
    // Use multithreading to parallelize the operation
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 threads if detection fails
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    for (unsigned int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        
        threads.push_back(std::thread(
            identifyConnectedComponentsChunk, terrain, labels, width, height, startY, endY
        ));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

void propagateLabelsChunk(int* terrain, int* labels, int width, int height, bool* changed, int startY, int endY) {
    bool localChanged = false;
    
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int currentType = terrain[idx];
            int currentLabel = labels[idx];
            
            // Check 4-connected neighbors
            int dx[4] = {-1, 1, 0, 0};
            int dy[4] = {0, 0, -1, 1};
            
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nidx = ny * width + nx;
                    
                    // If same terrain type and smaller label, adopt it
                    if (terrain[nidx] == currentType && labels[nidx] < currentLabel) {
                        labels[idx] = labels[nidx];
                        localChanged = true;
                        break;
                    }
                }
            }
        }
    }
    
    // If any thread found a change, set the global changed flag
    if (localChanged) {
        *changed = true;
    }
}

void propagateLabels(int* terrain, int* labels, int width, int height, bool* changed) {
    *changed = false;
    
    // Use multithreading to parallelize the operation
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 threads if detection fails
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    for (unsigned int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        
        threads.push_back(std::thread(
            propagateLabelsChunk, terrain, labels, width, height, changed, startY, endY
        ));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

void countComponentSizes(int* labels, int* componentSizes, int width, int height) {
    // Initialize component sizes to 0
    // Note: We need to allocate space for all possible labels (width*height)
    // since each pixel could potentially have its own unique label
    memset(componentSizes, 0, width * height * sizeof(int));
    
    // Count the number of pixels in each component
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int label = labels[idx];
            componentSizes[label]++;
        }
    }
}

void removeSmallComponentsChunk(int* terrain, int* labels, int* output, int* componentSizes, int minSize, int width, int height, int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int label = labels[idx];
            
            if (componentSizes[label] < minSize) {
                // Replace small components with the most common neighbor type
                int typeCounts[31] = {0};
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int nidx = ny * width + nx;
                            
                            // Only count neighboring cells from different components
                            if (labels[nidx] != label) {
                                typeCounts[terrain[nidx]]++;
                            }
                        }
                    }
                }
                
                // Find most common neighboring type
                int bestType = terrain[idx];
                int maxCount = 0;
                
                for (int t = 0; t < 31; t++) {
                    if (typeCounts[t] > maxCount) {
                        maxCount = typeCounts[t];
                        bestType = t;
                    }
                }
                
                output[idx] = bestType;
            } else {
                // Keep original terrain for larger components
                output[idx] = terrain[idx];
            }
        }
    }
}

void removeSmallComponents(int* terrain, int* labels, int* output, int* componentSizes, int minSize, int width, int height) {
    // First, count the size of each component
    countComponentSizes(labels, componentSizes, width, height);
    
    // Use multithreading to parallelize the operation
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 threads if detection fails
    
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    
    for (unsigned int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        
        threads.push_back(std::thread(
            removeSmallComponentsChunk, terrain, labels, output, componentSizes, minSize, width, height, startY, endY
        ));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}