#include "../include/terrain_gen.h"
#include "../include/terrain_types.h"
#include "../include/perlin_noise.h"

__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Generate noise value
        float nx = (float)x / width * scale + offsetX;
        float ny = (float)y / height * scale + offsetY;
        // Fixed z value for 2D terrain
        float nz = 0.0f;
        
        // Get noise value between -1 and 1
        float value = noise(nx, ny, nz);
        
        // Scale to 0 to 1 range
        value = (value + 1.0f) * 0.5f;
        
        // Map noise to terrain types (simple example)
        int terrainType;
        if (value < 0.2f) {
            terrainType = WATER;
        } else if (value < 0.3f) {
            terrainType = SAND;
        } else if (value < 0.7f) {
            terrainType = GRASS;
        } else if (value < 0.8f) {
            terrainType = ROCK;
        } else {
            terrainType = SNOW;
        }
        
        // Store terrain type in output array
        terrain[y * width + x] = terrainType;
    }
}

void createPerlinNoiseTerrain(int* d_terrain, int width, int height, 
                              float scale, float offsetX, float offsetY) {

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    generateTerrain<<<gridSize, blockSize>>>(d_terrain, width, height, scale, offsetX, offsetY);
}