#include "../../include/terrain/component_analysis.h"

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
    *changed = false; 
    
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