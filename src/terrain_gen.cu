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

__device__ float distributedNoise(float x, float y, float z, int octaves) {
    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maxValue = 0.0f;
    
    for(int i = 0; i < octaves; i++) {
        total += enhancedNoise(x * frequency, y * frequency, z) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.6f;  // slower decay
        frequency *= 1.9f;  // prime-based multiplier instead of 2.0f
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
    float secondMinDist = 1000.0f;  // Track second closest point too
    
    // Check a wider area of cells - increase search radius
    for (int i = -1; i <= 1; i++) {  // Was -1 to 1
        for (int j = -1; j <= 1; j++) {  // Was -1 to 1
            // Get the neighboring cell coordinates
            int xj = xi + i;
            int yj = yi + j;
            
            // Use better hash function with larger primes
            int hash = ((xj * 12721 + yj * 31337 + seed * 6971) & 0x7fffffff);
            float hashX = (hash % 1024) / 1024.0f;
            float hashY = ((hash / 1024) % 1024) / 1024.0f;
            
            // Position of feature point in this cell
            float px = xj + hashX;
            float py = yj + hashY;
            
            // Distance to feature point
            float dx = (px * cellSize) - x;
            float dy = (py * cellSize) - y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            // Update min and second min distances
            if (dist < minDist) {
                secondMinDist = minDist;
                minDist = dist;
            } else if (dist < secondMinDist) {
                secondMinDist = dist;
            }
        }
    }
    
    // Return the difference between closest and second closest
    // This creates more natural cell boundaries
    return secondMinDist - minDist;
}

// Improved terrain generation with modulo approach to ensure all terrain types appear
__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Coordinates for various noise functions
        float nx = (float)x / width * scale + offsetX;
        float ny = (float)y / height * scale + offsetY;
        
        // Domain warping - distort the space to break up the pattern
        float warpStrength = 0.2f;
        float warpX = nx + warpStrength * distributedNoise(nx * 2.0f, ny * 2.0f, 0.0f, 4);
        float warpY = ny + warpStrength * distributedNoise(nx * 2.0f + 100.0f, ny * 2.0f + 100.0f, 0.0f, 4);
        
        // Use warped coordinates for noise generation
        float noise1 = distributedNoise(warpX, warpY, 0.0f, 6);
        float noise2 = distributedNoise(warpX * 2.3f, warpY * 1.7f, 0.5f, 4);
        float noise3 = distributedNoise(warpX * 5.1f, warpY * 4.7f, 1.0f, 3);
        float noise4 = distributedNoise(warpX * 11.3f, warpY * 7.9f, 2.0f, 2);
        
        // Combine noise layers
        float elevation = 0.5f * noise1 + 0.25f * noise2 + 0.125f * noise3 + 0.125f * noise4;
        float moisture = distributedNoise(warpX + 100.0f, warpY + 100.0f, 1.0f, 4);
        
        // Variable scale for Voronoi to create different sized regions
        float variableScale = scale * (0.8f + 0.4f * noise(nx * 0.01f, ny * 0.01f, 0.5f));
        float cellScale = 0.05f * variableScale;
        
        // Apply domain warping to Voronoi coordinates too
        float vwarpX = (float)x / width * cellScale + offsetX * 0.1f + 0.3f * noise(nx * 0.5f, ny * 0.5f, 0.0f);
        float vwarpY = (float)y / height * cellScale + offsetY * 0.1f + 0.3f * noise(nx * 0.5f + 50.0f, ny * 0.5f + 50.0f, 0.0f);
        
        // Multiple Voronoi layers with different frequencies
        float v1 = voronoiNoise(vwarpX, vwarpY, 12345);
        float v2 = voronoiNoise(vwarpX * 2.3f, vwarpY * 2.3f, 54321);
        float v3 = voronoiNoise(vwarpX * 0.5f, vwarpY * 0.5f, 98765);
        float voronoiValue = v1 * 0.5f + v2 * 0.3f + v3 * 0.2f;
        
        // Get local variation with domain warping
        float localScale = scale * 0.5f;
        float lwarpX = (float)x / width * localScale + offsetX + 500.0f + 0.2f * noise(nx * 1.5f, ny * 1.5f, 0.0f);
        float lwarpY = (float)y / height * localScale + offsetY + 500.0f + 0.2f * noise(nx * 1.5f + 200.0f, ny * 1.5f + 200.0f, 0.0f);
        float localVar = distributedNoise(lwarpX, lwarpY, 2.0f, 3);
        
        // Create a more natural biome selector with more variation and less pattern
        float biomeSelector = (voronoiValue * 1.5f + elevation * 0.5f + moisture * 0.5f + localVar * 0.3f);
        biomeSelector = biomeSelector * (0.8f + 0.4f * noise(nx * 7.9f, ny * 11.3f, 0.0f));
        
        // Variable region frequency to break up the pattern
        float regionFreq = 7.0f + noise(nx * 0.1f, ny * 0.1f, 0.0f) * 3.0f;
        int regionType = (int)(biomeSelector * regionFreq) % 6;
        
        // Rest of your code (terrain type assignment) remains the same
        int terrainType;

        float desertVar = localVar + 0.1f * noise(nx * 13.7f, ny * 17.3f, 0.0f); // Add micro-variation

        // Assign terrain based on region type with local variations
        switch(regionType) {
            case 0: // Desert regions
                if (desertVar < 0.3f) terrainType = DESERT;
                else if (desertVar < 0.6f) terrainType = SAND;
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