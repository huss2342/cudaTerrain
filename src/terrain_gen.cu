#include "../include/terrain_gen.h"
#include "../include/terrain_types.h"
#include "../include/perlin_noise.h"
#include <math.h>

// Enhanced noise function with better distribution of values
__device__ float enhancedNoise(float x, float y, float z) {
    float val = noise(x, y, z);
    val = (val + 1.0f) * 0.5f;
    // Use a milder transformation
    val = powf(val, 0.9f);
    return val;
}

// Multi-octave noise with forced distribution
__device__ float distributedNoise(float x, float y, float z, int octaves) {
    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maxValue = 0.0f;
    
    for(int i = 0; i < octaves; i++) {
        total += enhancedNoise(x * frequency, y * frequency, z) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.6f;  // slower decay
        frequency *= 2.0f;
    }
    
    total /= maxValue;
    return total;
}

// Simple 2D Voronoi noise implementation
__device__ float voronoiNoise(float x, float y, int seed) {
    // Define a grid cell size
    const float cellSize = 1.0f;
    
    // Find the integer coordinates of the cell
    int xi = floorf(x / cellSize);
    int yi = floorf(y / cellSize);
    
    float minDist = 1000.0f;
    
    // Check the current cell and surrounding cells
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            // Get the neighboring cell coordinates
            int xj = xi + i;
            int yj = yi + j;
            
            // Generate a pseudo-random position within this cell
            // Using a simple hash function
            int hash = (xj * 13 + yj * 17 + seed) & 0x7fffffff;
            float hashX = (hash % 1024) / 1024.0f; // Random 0-1
            float hashY = ((hash / 1024) % 1024) / 1024.0f; // Random 0-1
            
            // Position of feature point in this cell
            float px = xj + hashX;
            float py = yj + hashY;
            
            // Distance to feature point
            float dx = px - x / cellSize;
            float dy = py - y / cellSize;
            float dist = sqrtf(dx*dx + dy*dy);
            
            // Keep track of minimum distance
            minDist = fminf(minDist, dist);
        }
    }
    
    return minDist;
}

// Improved terrain generation with modulo approach to ensure all terrain types appear
__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Coordinates for various noise functions
        float nx = (float)x / width * scale + offsetX;
        float ny = (float)y / height * scale + offsetY;
        
        // Generate base values with Perlin noise
        float elevation = distributedNoise(nx, ny, 0.0f, 6);
        float moisture = distributedNoise(nx + 100.0f, ny + 100.0f, 1.0f, 4);
        
        // Use Voronoi for region definition (cell-like regions)
        // Scale down for larger regions
        float cellScale = 0.05f * scale;
        float vnx = (float)x / width * cellScale + offsetX * 0.1f;
        float vny = (float)y / height * cellScale + offsetY * 0.1f;
        
        // Multiple Voronoi layers for more interesting shapes
        float v1 = voronoiNoise(vnx, vny, 12345);
        float v2 = voronoiNoise(vnx * 2.0f, vny * 2.0f, 54321);
        float voronoiValue = v1 * 0.7f + v2 * 0.3f;
        
        // Get local variation within regions
        float localScale = scale * 0.5f;
        float lx = (float)x / width * localScale + offsetX + 500.0f;
        float ly = (float)y / height * localScale + offsetY + 500.0f;
        float localVar = distributedNoise(lx, ly, 2.0f, 3);
        
        // Combine values to determine biome regions
        // This creates more organic boundaries
        float biomeSelector = (voronoiValue * 2.0f + elevation * 0.3f + moisture * 0.3f);
        
        // Map to region type (0-5)
        int regionType = (int)(biomeSelector * 6.0f) % 6;
        
        // Define terrain type based on region with local variations
        int terrainType;
        
        // Assign terrain based on region type with local variations
        switch(regionType) {
            case 0: // Desert regions
                if (localVar < 0.3f) terrainType = DESERT;
                else if (localVar < 0.6f) terrainType = SAND;
                else terrainType = DUNE;
                break;
                
            case 1: // Forest regions
                if (localVar < 0.25f) terrainType = FOREST;
                else if (localVar < 0.5f) terrainType = GRASS;
                else if (localVar < 0.75f) terrainType = JUNGLE;
                else terrainType = TAIGA;
                break;
                
            case 2: // Mountain regions
                if (elevation > 0.7f) {
                    if (localVar < 0.5f) terrainType = MOUNTAIN;
                    else terrainType = CLIFF;
                } else {
                    if (localVar < 0.5f) terrainType = ROCK;
                    else terrainType = PLATEAU;
                }
                break;
                
            case 3: // Tundra regions
                if (localVar < 0.3f) terrainType = TUNDRA;
                else if (localVar < 0.6f) terrainType = SNOW;
                else terrainType = GLACIER;
                break;
                
            case 4: // Grassland regions
                if (localVar < 0.25f) terrainType = GRASS;
                else if (localVar < 0.5f) terrainType = PRAIRIE;
                else if (localVar < 0.75f) terrainType = STEPPE;
                else terrainType = SAVANNA;
                break;
                
            case 5: // Water regions
                if (localVar < 0.3f) terrainType = WATER;
                else if (localVar < 0.5f) terrainType = BAY;
                else if (localVar < 0.7f) terrainType = FJORD;
                else terrainType = COVE;
                break;
                
            default:
                terrainType = GRASS; // Fallback
        }
        
        // Override for extreme elevations regardless of region
        if (elevation > 0.9f) {
            if (localVar < 0.3f) terrainType = MOUNTAIN;
            else if (localVar < 0.6f) terrainType = SNOW;
            else terrainType = GLACIER;
        }
        else if (elevation < 0.1f) {
            if (localVar < 0.3f) terrainType = WATER;
            else if (localVar < 0.6f) terrainType = BAY;
            else terrainType = COVE;
        }
        
        terrain[y * width + x] = terrainType;
    }
}
__global__ void smoothTerrain(int* terrain, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Skip edges
        if (x == 0 || y == 0 || x == width-1 || y == height-1) {
            output[idx] = terrain[idx];
            return;
        }
        
        // Count occurrences of neighboring terrain types
        int typeCounts[31] = {0};
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int nidx = ny * width + nx;
                typeCounts[terrain[nidx]]++;
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
        
        // Assign most common type
        output[idx] = bestType;
    }
}

void createPerlinNoiseTerrain(int* d_terrain, int width, int height,
    float scale, float offsetX, float offsetY) {
    // Normalize scale
    float adjustedScale = fmaxf(0.05f, fminf(fabs(scale), 100.0f));

    // Block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Generate initial terrain
    generateTerrain<<<gridSize, blockSize>>>(d_terrain, width, height, adjustedScale, offsetX, offsetY);

    // Allocate temporary buffer for smoothing
    int* d_tempTerrain;
    cudaMalloc(&d_tempTerrain, width * height * sizeof(int));

    // Apply multiple smoothing passes (2-3 passes work well)
    for (int i = 0; i < 3; i++) {
    // Smooth from d_terrain to d_tempTerrain
    smoothTerrain<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height);

    // Copy back to d_terrain for next iteration
    cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Clean up
    cudaFree(d_tempTerrain);
}