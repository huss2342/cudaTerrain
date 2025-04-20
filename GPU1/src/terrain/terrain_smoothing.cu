#include "../../include/terrain/terrain_smoothing.h"
#include <math.h>

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
        
        const int filterSize = 500;
        const int halfFilterSize = filterSize / 2;

        // Use a larger neighborhood to better determine if this is an isolated patch
        for (int dy = -halfFilterSize; dy <= halfFilterSize; dy++) {
            for (int dx = -halfFilterSize; dx <= halfFilterSize; dx++) {
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