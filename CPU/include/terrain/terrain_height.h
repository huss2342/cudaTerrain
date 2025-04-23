#ifndef TERRAIN_HEIGHT_H
#define TERRAIN_HEIGHT_H

// Height generation and manipulation functions
void generateHeightMap(int* terrain, float* heightMap, int width, int height, 
                      float scale, float offsetX, float offsetY);

// Erosion simulation
void simulateErosion(float* heightMap, float* output, int width, int height,
                    int iterations, float erosionRate);

// Utility functions for height operations
float getTerrainBaseHeight(int terrainType);
float blendHeight(float baseHeight, float noiseValue);

#endif // TERRAIN_HEIGHT_H