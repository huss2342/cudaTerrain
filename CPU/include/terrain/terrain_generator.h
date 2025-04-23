// CPU/include/terrain/terrain_generator.h
#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

// Core terrain generation function
void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY);

// Main function to create terrain
void createPerlinNoiseTerrain(int* terrain, int width, int height, 
                             float scale = 8.0f, float offsetX = 0.0f, 
                             float offsetY = 0.0f);

#endif // TERRAIN_GENERATOR_H