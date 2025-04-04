#include "../include/connectivity.h"
#include "../include/terrain_types.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <queue>
#include <vector>
#include <algorithm>

// Function to expand existing walkable areas
void expandWalkableAreas(int* d_terrain, int width, int height) {
    printf("Expanding existing walkable areas...\n");
    
    // Allocate host memory for terrain modification
    int* h_terrain = new int[width * height];
    int* h_output = new int[width * height];
    
    // Copy terrain data to host
    cudaMemcpy(h_terrain, d_terrain, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(h_output, h_terrain, width * height * sizeof(int));
    
    // Define the most important walkable terrain types
    const int primaryWalkables[] = {GRASS, FOREST, 
                                    //PRAIRIE, STEPPE, SAND, BEACH
                                    };
    const int numPrimaryTypes = sizeof(primaryWalkables) / sizeof(primaryWalkables[0]);
    
    // Iteration count - how many expansion passes to perform
    const int expansionPasses = 5;
    
    // For each pass, expand outward
    for (int pass = 0; pass < expansionPasses; pass++) {
        printf("Expansion pass %d/%d...\n", pass + 1, expansionPasses);
        
        // For each pixel
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                int terrainType = h_terrain[idx];
                
                // Skip if this is already a walkable terrain
                const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
                if (terrainInfo && terrainInfo->walkable) {
                    continue;
                }
                
                // Check neighbors (8-connected)
                static const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
                static const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
                
                // Count occurrences of each primary terrain type in neighbors
                int typeCounts[numPrimaryTypes] = {0};
                
                for (int d = 0; d < 8; d++) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int nidx = ny * width + nx;
                        int neighborType = h_terrain[nidx];
                        
                        // Count this neighbor's type
                        for (int t = 0; t < numPrimaryTypes; t++) {
                            if (neighborType == primaryWalkables[t]) {
                                typeCounts[t]++;
                                break;
                            }
                        }
                    }
                }
                
                // Find which walkable type has the most neighbors
                int bestTypeIdx = -1;
                int maxCount = 0;
                
                for (int t = 0; t < numPrimaryTypes; t++) {
                    if (typeCounts[t] > maxCount) {
                        maxCount = typeCounts[t];
                        bestTypeIdx = t;
                    }
                }
                
                // If there's at least one walkable neighbor, expand it
                if (bestTypeIdx >= 0 && maxCount >= 1) {
                    h_output[idx] = primaryWalkables[bestTypeIdx];
                }
            }
        }
        
        // Copy expanded terrain for next pass
        memcpy(h_terrain, h_output, width * height * sizeof(int));
    }
    
    // Copy modified terrain back to device
    cudaMemcpy(d_terrain, h_output, width * height * sizeof(int), cudaMemcpyHostToDevice);
    
    delete[] h_terrain;
    delete[] h_output;
    printf("Walkable area expansion completed\n");
}

// Main connectivity function now uses expansion instead of paths
void connectLandmasses(int* d_terrain, int width, int height) {
    printf("Starting area expansion for better playability...\n");
    
    // Simply expand the existing walkable areas
    expandWalkableAreas(d_terrain, width, height);
    
    printf("Area expansion completed\n");
}