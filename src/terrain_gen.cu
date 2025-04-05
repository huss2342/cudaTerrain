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
    float cellSize = 1.0f;
    
    // Normalize coordinates
    float nx = x / cellSize;
    float ny = y / cellSize;
    
    // Get integer cell coordinates
    int xi = floorf(nx);
    int yi = floorf(ny);
    
    // Get fractional part (position within cell)
    // float xf = nx - xi;
    // float yf = ny - yi;
    
    float minDist = 1000.0f;
    float secondMinDist = 1000.0f;
    
    // Check surrounding cells
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int cx = xi + i;
            int cy = yi + j;
            
            // Get random position in cell using improved hash
            unsigned int h = seed;
            h = (h * 747796405) + cx;
            h = (h * 747796405) + cy;
            h = (h ^ (h >> 12)) * 1664525;
            h = (h ^ (h >> 19)) * 1013904223;
            float fx = (h & 0xFFFF) / 65536.0f;
            
            h = (h * 747796405) + cx;
            h = (h * 747796405) + cy;
            h = (h ^ (h >> 12)) * 1664525;
            h = (h ^ (h >> 19)) * 1013904223;
            float fy = (h & 0xFFFF) / 65536.0f;
            
            // Feature point position (in normalized space)
            float px = cx + fx;
            float py = cy + fy;
            
            // Distance to feature point (using explicit parentheses to be clear)
            float dx = px - nx;
            float dy = py - ny;
            float dist = sqrtf((dx*dx) + (dy*dy));
            
            // Update distances
            if (dist < minDist) {
                secondMinDist = minDist;
                minDist = dist;
            } else if (dist < secondMinDist) {
                secondMinDist = dist;
            }
        }
    }
    
    // Return difference between distances as cell boundary
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

        float randVal = noise(nx * 23.4f, ny * 19.7f, 0.7f);
        
        // Variable region frequency to break up the pattern
        float regionFreq = 7.0f + noise(nx * 0.1f, ny * 0.1f, 0.0f) * 3.0f;
        int regionType = (int)(biomeSelector * regionFreq) % 6;

        const float waterChance = 0.15f;
        const int maxWaterRerolls = 5;
        for (int i = 0; i < maxWaterRerolls; ++i) {
            if (regionType != 5) break; // Not water — stop checking
            float randVal = 0.5f + 0.5f * noise(nx * 83.1f + i * 17.31f, ny * 47.2f + i * 9.13f, 1.23f);
            if (randVal < waterChance) break;
            regionType = (int)(biomeSelector * regionFreq) % 6;
        }        
        

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
                if (localVar < 0.15f) terrainType = WATER;
                else if (localVar < 0.3f) terrainType = BAY;
                else if (localVar < 0.5f) terrainType = FJORD;
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
        if (x <= 1 || y <= 1 || x >= width-2 || y >= height-2) {
            output[idx] = terrain[idx];
            return;
        }
        
        // Count occurrences of neighboring terrain types with wider horizontal sampling
        int typeCounts[31] = {0};
        
        // Use a wider horizontal neighborhood to combat vertical striping
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -2; dx <= 2; dx++) {  // Wider in x-direction
                int nx = x + dx;
                if (nx < 0) nx = 0;
                if (nx >= width) nx = width-1;
                
                int ny = y + dy;
                int nidx = ny * width + nx;
                
                // Weight horizontal neighbors more heavily
                int weight = (abs(dx) <= 1) ? 1 : 1;  // Equal weighting for all neighbors
                typeCounts[terrain[nidx]] += weight;
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

__global__ void removeVerticalStripes(int* terrain, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Skip edges
        if (x <= 2 || x >= width-3) {
            output[idx] = terrain[idx];
            return;
        }
        
        // Look at horizontal neighbors only
        int currentType = terrain[idx];
        int leftType1 = terrain[idx-1];
        int leftType2 = terrain[idx-2];
        int rightType1 = terrain[idx+1];
        int rightType2 = terrain[idx+2];
        
        // If current pixel is different from BOTH neighbors on either side,
        // and those neighbors are the same, replace with the neighbor type
        if (leftType1 == leftType2 && rightType1 == rightType2 && 
            leftType1 != currentType && rightType1 != currentType) {
            // We're in a vertical stripe - replace with most common horizontal neighbor
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

__global__ void cleanupSmallPatches(int* terrain, int* output, int width, int height, int minRegionSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Skip the edges
        if (x <= 2 || y <= 2 || x >= width-3 || y >= height-3) {
            output[idx] = terrain[idx];
            return;
        }
        
        int currentType = terrain[idx];
        
        // Count how many adjacent cells have the same type
        int sameTypeCount = 0;
        int totalCount = 0;
        int neighborTypes[31] = {0}; // to track the most common neighbor type
        
        // Use a larger neighborhood to better determine if this is an isolated patch
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                // Skip the center point
                if (dx == 0 && dy == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                
                // Check bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                
                int nidx = ny * width + nx;
                int neighborType = terrain[nidx];
                
                totalCount++;
                neighborTypes[neighborType]++;
                
                if (neighborType == currentType) {
                    sameTypeCount++;
                }
            }
        }
        
        // If less than minRegionSize% of neighbors are the same type, 
        // this is likely an isolated patch or noise
        float sameRatio = (float)sameTypeCount / totalCount;
        
        if (sameRatio < (float)minRegionSize / 100.0f) {
            // Find the most common neighboring type to replace with
            int bestType = currentType;
            int maxCount = 0;
            
            for (int t = 0; t < 31; t++) {
                if (neighborTypes[t] > maxCount) {
                    maxCount = neighborTypes[t];
                    bestType = t;
                }
            }
            
            // Replace with the most common neighbor type
            output[idx] = bestType;
        } else {
            // Keep current type
            output[idx] = currentType;
        }
    }
}

__global__ void improvedSmoothTerrain(int* terrain, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Skip edges
        if (x <= 1 || y <= 1 || x >= width-2 || y >= height-2) {
            output[idx] = terrain[idx];
            return;
        }
        
        int currentType = terrain[idx];
        // Count occurrences of neighboring terrain types
        int typeCounts[31] = {0};
        
        // Check a wider radius for better smoothing
        const int radius = 30;
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                // Skip out-of-bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                
                int nidx = ny * width + nx;
                int neighborType = terrain[nidx];
                
                // Weight by distance and bias toward preserving the current type
                float dist = sqrtf((float)(dx*dx + dy*dy));
                int weight;
                
                if (neighborType == currentType) {
                    // Heavily favor keeping the current type to prevent over-smoothing
                    weight = 3 + (radius - (int)dist);
                } else {
                    // Other types get less weight with distance
                    weight = radius - (int)dist;
                    if (weight <= 0) weight = 1;
                }
                
                typeCounts[neighborType] += weight;
            }
        }
        
        // Find the type with highest count (weighted)
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

__global__ void removeIsolatedNoise(int* terrain, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Skip edges
        if (x <= 1 || y <= 1 || x >= width-2 || y >= height-2) {
            output[idx] = terrain[idx];
            return;
        }
        
        int currentType = terrain[idx];
        
        // Check immediate neighbors only (3x3 area)
        int sameTypeCount = 0;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue; // Skip center
                
                int nx = x + dx;
                int ny = y + dy;
                int nidx = ny * width + nx;
                
                if (terrain[nidx] == currentType) {
                    sameTypeCount++;
                }
            }
        }
        
        // If this is a single isolated pixel (no same neighbors), replace it
        if (sameTypeCount == 0) {
            // Find the most common neighbor type
            int typeCounts[31] = {0};
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    int nidx = ny * width + nx;
                    
                    typeCounts[terrain[nidx]]++;
                }
            }
            
            // Find most common type
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
        // For small clusters (1-2 same neighbors), we could also replace them
        else if (sameTypeCount <= 2) {
            // Similar replacement logic as above
            int typeCounts[31] = {0};
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    int nidx = ny * width + nx;
                    
                    // Don't count pixels of the same type to avoid bias
                    if (terrain[nidx] != currentType) {
                        typeCounts[terrain[nidx]]++;
                    }
                }
            }
            
            // Find most common surrounding type (not counting current type)
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
            // Keep original type for larger clusters
            output[idx] = currentType;
        }
    }
}


__global__ void identifyConnectedComponents(int* terrain, int* labels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
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

__global__ void propagateLabels(int* terrain, int* labels, int width, int height, bool* changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
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
                    *changed = true;
                    break;
                }
            }
        }
    }
}

__global__ void removeSmallComponents(int* terrain, int* labels, int* output, int* componentSizes, int minSize, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
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

void createPerlinNoiseTerrain(int* d_terrain, int width, int height,
    float scale, float offsetX, float offsetY) {
    // Normalize scale
    float adjustedScale = fmaxf(0.05f, fminf(fabs(scale), 100.0f));
    
    // Block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Generate initial terrain
    generateTerrain<<<gridSize, blockSize>>>(d_terrain, width, height, adjustedScale, offsetX, offsetY);

    // Allocate temporary buffers
    int* d_tempTerrain;
    cudaMalloc(&d_tempTerrain, width * height * sizeof(int));
    
    // First, remove isolated noise (single pixels or very small clusters)
    for (int i = 0; i < 5; i++) {
        removeIsolatedNoise<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height);
        cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    // Apply a light smoothing with a small radius
    improvedSmoothTerrain<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height);
    cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // Remove vertical striping artifacts
    removeVerticalStripes<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height);
    cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // Use a more aggressive cleanup but with a smaller threshold
    cleanupSmallPatches<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height, 20);
    cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // Final light smoothing
    smoothTerrain<<<gridSize, blockSize>>>(d_terrain, d_tempTerrain, width, height);
    cudaMemcpy(d_terrain, d_tempTerrain, width * height * sizeof(int), cudaMemcpyDeviceToDevice);

    // Clean up
    cudaFree(d_tempTerrain);
}