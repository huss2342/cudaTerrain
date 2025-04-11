#include "../../include/visualization/visualization.h"
#include "../../include/terrain/terrain_types.h"
#include <stdio.h>

__global__ void visualizeTerrain(int* terrain, unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int terrainType = terrain[idx];
        
        // Get the color from the terrain type
        const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
        
        // Set RGB values in image (assuming 3 channels)
        image[idx * 3 + 0] = terrainInfo->color.r;
        image[idx * 3 + 1] = terrainInfo->color.g;
        image[idx * 3 + 2] = terrainInfo->color.b;
    }
}

void saveToPPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(image, 3, width * height, fp);
    
    fclose(fp);
    printf("Saved terrain image to %s\n", filename);
}