#include "../include/terrain_gen.h"
#include "../include/terrain_types.h"
#include "../include/perlin_noise.h"
#include <math.h>

// Enhanced noise function with better distribution of values
__device__ float enhancedNoise(float x, float y, float z) {
    // Get base Perlin noise value
    float val = noise(x, y, z);
    
    // Apply non-linear transformation to expand the "interesting" range
    // This emphasizes the mid-range values where most terrain variation happens
    val = (val + 1.0f) * 0.5f; // Convert from [-1,1] to [0,1]
    val = powf(val, 0.8f);     // Apply slight curve to distribute values better
    
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
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    
    // Normalize to [0,1] range
    total /= maxValue;
    
    return total;
}

// Improved terrain generation with modulo approach to ensure all terrain types appear
__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Base coordinates - use higher frequencies for more detail
        float nx = (float)x / width * scale + offsetX;
        float ny = (float)y / height * scale + offsetY;
        
        // Generate multiple noise layers with different seeds for variety
        float elevation = distributedNoise(nx, ny, 0.0f, 8); // Increased octaves for more detail
        float moisture = distributedNoise(nx + 100.0f, ny + 100.0f, 1.0f, 6);
        float variation = distributedNoise(nx + 200.0f, ny + 200.0f, 2.0f, 4);
        
        // Use modulo approach to ensure all terrain types appear
        // Create a hash value from the noise values to get distribution across all terrain types
        float hash = (elevation * 13.0f + moisture * 17.0f + variation * 19.0f) * 100.0f;
        int hashInt = (int)hash;
        
        // Use abs() to handle negative values and modulo to ensure we get all terrain types
        int terrainType = abs(hashInt % 31); // 31 is the total number of terrain types
        
        // Add some clustering to make the terrain less random/noisy
        // Higher elevation areas tend to be peaks/mountains
        if (elevation > 0.85f) {
            // High elevation areas
            if (hashInt % 4 == 0) {
                terrainType = MOUNTAIN;
            } else if (hashInt % 4 == 1) {
                terrainType = SNOW;
            } else if (hashInt % 4 == 2) {
                terrainType = GLACIER;
            } else {
                terrainType = CLIFF;
            }
        }
        // Water areas for lower elevation
        else if (elevation < 0.15f) {
            // Low elevation areas
            if (hashInt % 4 == 0) {
                terrainType = WATER;
            } else if (hashInt % 4 == 1) {
                terrainType = BAY;
            } else if (hashInt % 4 == 2) {
                terrainType = FJORD;
            } else {
                terrainType = COVE;
            }
        }
        // Desert areas for dry regions
        else if (elevation < 0.4f && moisture < 0.3f) {
            if (hashInt % 3 == 0) {
                terrainType = DESERT;
            } else if (hashInt % 3 == 1) {
                terrainType = SAND;
            } else {
                terrainType = DUNE;
            }
        }
        // For areas that don't match specific conditions, use the modulo approach
        
        // Store terrain type
        terrain[y * width + x] = terrainType;
    }
}

void createPerlinNoiseTerrain(int* d_terrain, int width, int height, 
                             float scale, float offsetX, float offsetY) {
    // Normalize scale to a reasonable range - decreased min scale for larger features in big maps
    float adjustedScale = fmaxf(0.05f, fminf(fabs(scale), 100.0f));
    
    // Use smaller block size for better occupancy with large maps
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    generateTerrain<<<gridSize, blockSize>>>(d_terrain, width, height, adjustedScale, offsetX, offsetY);
}