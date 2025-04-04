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

// CUDA kernel to identify walkable terrain
__global__ void identifyWalkableTerrain(int* terrain, int* walkable, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int terrainType = terrain[idx];
        
        // Check if terrain is walkable (not water and not steep)
        if (terrainType != WATER && terrainType != BAY && 
            terrainType != FJORD && terrainType != COVE &&
            terrainType != MOUNTAIN && terrainType != CLIFF && 
            terrainType != VOLCANO && terrainType != GLACIER &&
            terrainType != MESA && terrainType != CANYON) {
            walkable[idx] = 1; // Walkable
        } else {
            walkable[idx] = 0; // Not walkable
        }
    }
}


// Create a more comprehensive connectivity network
void createConnectivityNetwork(int* d_terrain, int* d_regions, int width, int height, int maxRegion) {
    // Create a grid of major "highways" across the map
    int gridSpacing = width / 8; // 8 vertical and horizontal paths
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Create horizontal and vertical paths at regular intervals
    for (int i = 1; i <= 7; i++) {
        int x = i * gridSpacing;
        createVerticalPath<<<gridSize, blockSize>>>(d_terrain, d_regions, width, height, x);
    }
    
    for (int j = 1; j <= 7; j++) {
        int y = j * gridSpacing;
        createHorizontalPath<<<gridSize, blockSize>>>(d_terrain, d_regions, width, height, y);
    }
}

// Kernel to create a vertical path
__global__ void createVerticalPath(int* terrain, int* regions, int width, int height, int x) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (threadX < width && threadY < height) {
        // Define path width
        int pathWidth = 40;
        
        // Check if this point is on the vertical path
        if (abs(threadX - x) <= pathWidth) {
            int idx = threadY * width + threadX;
            int terrainType = terrain[idx];
            
            // Only replace water
            if (terrainType == WATER || terrainType == BAY || 
                terrainType == FJORD || terrainType == COVE) {
                terrain[idx] = GRASS;
                regions[idx] = 1; // Assign to region 1
            }
        }
    }
}

// Kernel to create a horizontal path
__global__ void createHorizontalPath(int* terrain, int* regions, int width, int height, int y) {
    // Similar to createVerticalPath but for horizontal lines
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (threadX < width && threadY < height) {
        int pathWidth = 40;
        
        if (abs(threadY - y) <= pathWidth) {
            int idx = threadY * width + threadX;
            int terrainType = terrain[idx];
            
            if (terrainType == WATER || terrainType == BAY || 
                terrainType == FJORD || terrainType == COVE) {
                terrain[idx] = GRASS;
                regions[idx] = 1;
            }
        }
    }
}

// Function to perform flood fill to identify regions
void floodFill(int* walkable, int* regions, int width, int height) {
    // Copy data to host for CPU-based flood fill
    thrust::host_vector<int> h_walkable(width * height);
    thrust::host_vector<int> h_regions(width * height, 0);
    thrust::copy(walkable, walkable + width * height, h_walkable.begin());
    
    // Initialize variables
    int regionCount = 0;
    std::vector<std::pair<int, int>> dirs;
    dirs.push_back(std::make_pair(0, 1));
    dirs.push_back(std::make_pair(1, 0));
    dirs.push_back(std::make_pair(0, -1));
    dirs.push_back(std::make_pair(-1, 0));
    
    // Perform flood fill
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // If walkable and not assigned to a region yet
            if (h_walkable[idx] == 1 && h_regions[idx] == 0) {
                regionCount++;
                
                // BFS flood fill
                std::queue<std::pair<int, int>> queue;
                queue.push(std::make_pair(x, y));
                h_regions[idx] = regionCount;
                
                while (!queue.empty()) {
                    std::pair<int, int> front = queue.front();
                    int cx = front.first;
                    int cy = front.second;
                    queue.pop();
                    
                    // Check all 4 neighbors
                    for (size_t d = 0; d < dirs.size(); d++) {
                        int nx = cx + dirs[d].first;
                        int ny = cy + dirs[d].second;
                        
                        // Check bounds
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int nidx = ny * width + nx;
                            
                            // If walkable and not assigned to a region yet
                            if (h_walkable[nidx] == 1 && h_regions[nidx] == 0) {
                                h_regions[nidx] = regionCount;
                                queue.push(std::make_pair(nx, ny));
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Copy regions back to device
    thrust::copy(h_regions.begin(), h_regions.end(), regions);
}

// Kernel to create pathways between regions
__global__ void createPathways(int* terrain, int* regions, int width, int height, 
    int region1, int region2, int pathX1, int pathY1, 
    int pathX2, int pathY2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // Path width
        float pathWidth = 40.0f;

        // First create horizontal segment
        bool onHorizontalPath = (y >= fminf(pathY1, pathY2) - pathWidth && 
            y <= fmaxf(pathY1, pathY2) + pathWidth &&
            x >= fminf(pathX1, pathX2) - pathWidth && 
            x <= fmaxf(pathX1, pathX2) + pathWidth);

        // Check if this point is on the path
        if (onHorizontalPath) {
        // Only replace non-walkable terrain
        int terrainType = terrain[idx];
        if (terrainType == WATER || terrainType == BAY || 
            terrainType == FJORD || terrainType == COVE) {
                terrain[idx] = GRASS;
                regions[idx] = region1;
            }
        }
    }
}

// Function to find closest points between two regions
void findClosestPoints(int* regions, int width, int height, int region1, int region2,
                     int& x1, int& y1, int& x2, int& y2) {
    thrust::host_vector<int> h_regions(width * height);
    thrust::copy(regions, regions + width * height, h_regions.begin());
    
    std::vector<std::pair<int, int>> points1;
    std::vector<std::pair<int, int>> points2;
    
    // Collect points from each region
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (h_regions[idx] == region1) {
                points1.push_back(std::make_pair(x, y));
            } else if (h_regions[idx] == region2) {
                points2.push_back(std::make_pair(x, y));
            }
        }
    }
    
    // Find closest pair of points
    float minDist = INFINITY;
    
    for (size_t i = 0; i < points1.size(); i++) {
        const std::pair<int, int>& p1 = points1[i];
        for (size_t j = 0; j < points2.size(); j++) {
            const std::pair<int, int>& p2 = points2[j];
            float dist = sqrtf(powf(p1.first - p2.first, 2) + powf(p1.second - p2.second, 2));
            if (dist < minDist) {
                minDist = dist;
                x1 = p1.first;
                y1 = p1.second;
                x2 = p2.first;
                y2 = p2.second;
            }
        }
    }
}

// Main function to connect landmasses
void connectLandmasses(int* d_terrain, int width, int height) {
    // Allocate memory for walkable and region maps
    int* d_walkable;
    int* d_regions;
    cudaMalloc(&d_walkable, width * height * sizeof(int));
    cudaMalloc(&d_regions, width * height * sizeof(int));
    
    thrust::fill(thrust::device, d_regions, d_regions + width * height, 0);
    
    // CUDA kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Identify walkable terrain
    identifyWalkableTerrain<<<gridSize, blockSize>>>(d_terrain, d_walkable, width, height);
    
    // Perform flood fill to identify regions
    floodFill(d_walkable, d_regions, width, height);
    
    // Find the number of regions
    thrust::device_vector<int> d_regions_vec(d_regions, d_regions + width * height);
    int maxRegion = *thrust::max_element(d_regions_vec.begin(), d_regions_vec.end());
    
    printf("Found %d separate land regions\n", maxRegion);
    
    // If there's only one region or no regions, we're done
    if (maxRegion <= 1) {
        printf("No disconnected regions found, no paths needed\n");
        cudaFree(d_walkable);
        cudaFree(d_regions);
        return;
    }
    
    // Create a simple spanning tree to connect regions
    // For simplicity, we'll just connect region i to region i+1
    for (int i = 1; i < maxRegion; i++) {
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        
        // Find closest points between regions i and i+1
        findClosestPoints(d_regions, width, height, i, i+1, x1, y1, x2, y2);
        
        printf("Connecting region %d to region %d with path from (%d,%d) to (%d,%d)\n", 
               i, i+1, x1, y1, x2, y2);
        
        // Create pathway between the regions
        createPathways<<<gridSize, blockSize>>>(d_terrain, d_regions, width, height, 
                                              i, i+1, x1, y1, x2, y2);
    }
    
    // Clean up
    cudaFree(d_walkable);
    cudaFree(d_regions);
}