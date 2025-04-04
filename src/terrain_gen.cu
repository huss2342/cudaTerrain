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
    // Add variation to the cell size based on position
    float cellSize = 1.0f + enhancedNoise(x * 0.05f, y * 0.05f, seed * 0.01f) * 0.3f;
    
    // Perturb the coordinates slightly to break the grid pattern
    x += enhancedNoise(x * 0.1f, y * 0.07f, seed * 0.3f + 1.0f) * 0.5f;
    y += enhancedNoise(y * 0.1f, x * 0.07f, seed * 0.7f + 2.0f) * 0.5f;
    
    // Find the integer coordinates of the cell
    int xi = floorf(x / cellSize);
    int yi = floorf(y / cellSize);
    
    float minDist = 1000.0f;
    float secondMinDist = 1000.0f; // Track second closest for more interesting patterns
    
    // Check more surrounding cells (larger radius)
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            // Get the neighboring cell coordinates
            int xj = xi + i;
            int yj = yi + j;
            
            // More complex hash function with prime multipliers
            int hash = ((xj * 73856093) ^ (yj * 19349663) ^ seed) & 0x7fffffff;
            
            // Offset feature points away from cell centers
            float hashX = (hash % 1024) / 1024.0f;
            float hashY = ((hash / 1024) % 1024) / 1024.0f;
            
            // Add further perturbation based on cell position
            hashX += enhancedNoise(xj * 0.2f, yj * 0.3f, seed + 10) * 0.2f;
            hashY += enhancedNoise(yj * 0.2f, xj * 0.3f, seed + 20) * 0.2f;
            
            // Position of feature point in this cell
            float px = xj + hashX;
            float py = yj + hashY;
            
            // Distance to feature point
            float dx = px - x / cellSize;
            float dy = py - y / cellSize;
            float dist = sqrtf(dx*dx + dy*dy);
            
            // Keep track of minimum and second minimum distance
            if (dist < minDist) {
                secondMinDist = minDist;
                minDist = dist;
            } else if (dist < secondMinDist) {
                secondMinDist = dist;
            }
        }
    }
    
    // Return combination of distances for more varied patterns
    return minDist * 0.8f + (secondMinDist - minDist) * 0.2f;
}

// Improved terrain generation with modulo approach to ensure all terrain types appear
__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // int seed = 12345; // Fixed seed for reproducibility
    // int seedReverse = 54321; 

    // get seed from offsetX and offsetY
    const int seed = (int)(offsetX * 1000) + (int)(offsetY * 1000);
    const int seedReverse = (seed * 2654435761) ^ 0x5f3759df; // Uses golden ratio prime & xor trick

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
        float v1 = voronoiNoise(vnx, vny, seed);
        float v2 = voronoiNoise(vnx * 2.0f, vny * 2.0f, seedReverse);
        float v3 = voronoiNoise(vnx * 0.5f, vny * 0.5f, seed + 12345);
        float v4 = voronoiNoise(vnx * 1.7f + 100.0f, vny * 1.7f + 100.0f, seedReverse + 54321);
        
        float voronoiValue = v1 * 0.4f + v2 * 0.3f + v3 * 0.2f + v4 * 0.1f;

        // Add rotation to break grid pattern
        float angle = distributedNoise(nx * 0.007f, ny * 0.007f, 0.0f, 1) * 3.14159f;
        float cosA = cosf(angle);
        float sinA = sinf(angle);
        float rotX = vnx * cosA - vny * sinA;
        float rotY = vnx * sinA + vny * cosA;

        // Add rotated layer
        float vRot = voronoiNoise(rotX, rotY, seed + 31415);
        voronoiValue = voronoiValue * 0.8f + vRot * 0.2f;

        // Get local variation within regions
        float localScale = scale * 0.5f;
        float lx = (float)x / width * localScale + offsetX + 500.0f;
        float ly = (float)y / height * localScale + offsetY + 500.0f;
        float localVar = distributedNoise(lx, ly, 2.0f, 3);
        
        // Combine values to determine biome regions
        // This creates more organic boundaries
        float biomeSelector = (
            voronoiValue * 3.0f + 
            elevation * (0.2f + distributedNoise(nx * 2.7f, ny * 3.1f, 123.0f, 1) * 0.1f) + 
            moisture * (0.2f + distributedNoise(ny * 2.7f, nx * 3.1f, 321.0f, 1) * 0.1f) +
            distributedNoise(nx * 0.3f + ny * 0.2f, ny * 0.3f + nx * 0.2f, seed + 999.0f, 1) * 0.5f
        );

        // Use a more complex mapping to region types
        int regionHash = (int)(biomeSelector * 1000.0f);
        int regionType = abs(regionHash) % 6;
        
        // Add spatial variation to region selection
        float regionVariation = distributedNoise(nx * 0.03f, ny * 0.03f, seed + 444.0f, 1);
        if (regionVariation > 0.8f) {
            // Occasionally override for variety
            regionType = (regionType + (int)(regionVariation * 10.0f)) % 6;
        }

        // Define terrain type based on region with local variations
        int terrainType;

        // Add randomization to biome types based on position
        float biomeRandomizer = distributedNoise(nx * 0.15f, ny * 0.15f, seed + regionType * 1000.0f, 1);
        int biomePattern = (int)(biomeRandomizer * 10.0f) % 4; // 4 different patterns per biome type

        // Assign terrain based on region type with local variations
        switch(regionType) {
            case 0: // Desert regions
                if (biomePattern == 0) { 
                    // Standard mix
                    if (localVar < 0.3f) terrainType = DESERT;
                    else if (localVar < 0.6f) terrainType = SAND;
                    else terrainType = DUNE;
                } else if (biomePattern == 1) {
                    // Sand dominant
                    if (localVar < 0.2f) terrainType = DESERT;
                    else terrainType = SAND;
                } else if (biomePattern == 2) {
                    // Dune dominant
                    if (localVar < 0.25f) terrainType = DESERT;
                    else if (localVar < 0.4f) terrainType = SAND;
                    else terrainType = DUNE;
                } else {
                    // Desert/Dune mix
                    terrainType = (localVar < 0.55f) ? DESERT : DUNE;
                }
                break;
                
            case 1: // Forest regions
                if (biomePattern == 0) {
                    // Normal forest mix
                    if (localVar < 0.25f) terrainType = FOREST;
                    else if (localVar < 0.5f) terrainType = GRASS;
                    else if (localVar < 0.75f) terrainType = JUNGLE;
                    else terrainType = TAIGA;
                } else if (biomePattern == 1) {
                    // Temperate forest (no jungle)
                    if (localVar < 0.33f) terrainType = FOREST;
                    else if (localVar < 0.66f) terrainType = GRASS;
                    else terrainType = TAIGA;
                } else if (biomePattern == 2) {
                    // Tropical forest (no taiga)
                    if (localVar < 0.33f) terrainType = FOREST;
                    else if (localVar < 0.66f) terrainType = GRASS;
                    else terrainType = JUNGLE;
                } else {
                    // Dense forest (mostly forest and jungle)
                    if (localVar < 0.6f) terrainType = FOREST;
                    else terrainType = JUNGLE;
                }
                break;
                
            case 2: // Mountain regions
                if (biomePattern == 0) {
                    // Normal mountain mix
                    if (elevation > 0.7f) {
                        if (localVar < 0.5f) terrainType = MOUNTAIN;
                        else terrainType = CLIFF;
                    } else {
                        if (localVar < 0.5f) terrainType = ROCK;
                        else terrainType = PLATEAU;
                    }
                } else if (biomePattern == 1) {
                    // Rocky mountains (mostly rock and mountain)
                    if (localVar < 0.3f) terrainType = PLATEAU;
                    else if (localVar < 0.7f) terrainType = ROCK;
                    else terrainType = MOUNTAIN;
                } else if (biomePattern == 2) {
                    // Cliff dominant
                    if (elevation > 0.75f) {
                        terrainType = CLIFF;
                    } else {
                        terrainType = ROCK;
                    }
                } else {
                    // Plateau with mountains
                    if (elevation > 0.8f) {
                        terrainType = MOUNTAIN;
                    } else {
                        terrainType = PLATEAU;
                    }
                }
                break;
                
            case 3: // Tundra regions
                if (biomePattern == 0) {
                    // Normal tundra mix
                    if (localVar < 0.3f) terrainType = TUNDRA;
                    else if (localVar < 0.6f) terrainType = SNOW;
                    else terrainType = GLACIER;
                } else if (biomePattern == 1) {
                    // Snowy tundra
                    if (localVar < 0.25f) terrainType = TUNDRA;
                    else terrainType = SNOW;
                } else if (biomePattern == 2) {
                    // Icy tundra
                    if (localVar < 0.25f) terrainType = TUNDRA;
                    else if (localVar < 0.5f) terrainType = SNOW;
                    else terrainType = GLACIER;
                } else {
                    // Pure snow fields
                    terrainType = SNOW;
                }
                break;
                
            case 4: // Grassland regions
                if (biomePattern == 0) {
                    // Normal grassland mix
                    if (localVar < 0.25f) terrainType = GRASS;
                    else if (localVar < 0.5f) terrainType = PRAIRIE;
                    else if (localVar < 0.75f) terrainType = STEPPE;
                    else terrainType = SAVANNA;
                } else if (biomePattern == 1) {
                    // Prairie dominant
                    if (localVar < 0.3f) terrainType = GRASS;
                    else terrainType = PRAIRIE;
                } else if (biomePattern == 2) {
                    // Savanna dominant
                    if (localVar < 0.3f) terrainType = STEPPE;
                    else terrainType = SAVANNA;
                } else {
                    // Grassy steppes
                    if (localVar < 0.4f) terrainType = GRASS;
                    else terrainType = STEPPE;
                }
                break;
                
            case 5: // Water regions
                if (biomePattern == 0) {
                    // Full water mix
                    if (localVar < 0.3f) terrainType = WATER;
                    else if (localVar < 0.5f) terrainType = BAY;
                    else if (localVar < 0.7f) terrainType = FJORD;
                    else terrainType = COVE;
                } else if (biomePattern == 1) {
                    // Mostly open water
                    if (localVar < 0.8f) terrainType = WATER;
                    else terrainType = COVE;
                } else if (biomePattern == 2) {
                    // Bay and fjord mix
                    if (localVar < 0.4f) terrainType = WATER;
                    else if (localVar < 0.7f) terrainType = BAY;
                    else terrainType = FJORD;
                } else {
                    // Pure water
                    terrainType = WATER;
                }
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
                            float scale, float offsetX, float offsetY
) {
    // Normalize scale
    float adjustedScale = fmaxf(0.05f, fminf(fabs(scale), 100.0f));

    // Block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Generate initial terrain
    generateTerrain<<<gridSize, blockSize>>>(d_terrain, width, height, adjustedScale, offsetX, offsetY);

    // // Allocate temporary buffer for smoothing
    int* d_tempTerrain;
    cudaMalloc(&d_tempTerrain, width * height * sizeof(int));

    // Apply multiple smoothing passes
    for (int i = 0; i < 4; i++) {
        // Smooth from d_terrain to d_tempTerrain
        smoothTerrain<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height);

        // Swap pointers for next iteration
        int* temp = d_terrain;
        d_terrain = d_tempTerrain;
        d_tempTerrain = temp;
    }

    // If we did an even number of passes, the result is already in d_terrain
    // Otherwise, copy from d_tempTerrain back to d_terrain
    if (d_terrain != d_tempTerrain) {
        cudaFree(d_tempTerrain);
    } else {
        cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_tempTerrain);
    }
}