#ifndef VISUALIZATION_H
#define VISUALIZATION_H

// Kernel for terrain visualization
void visualizeTerrain(int* terrain, unsigned char* image, int width, int height);

// Updated function that includes height visualization
void visualizeTerrainWithHeight(int* terrain, float* heightMap, unsigned char* image, int width, int height);

// Function to save image to a PPM file
void saveToPPM(const char* filename, unsigned char* image, int width, int height);

#endif // VISUALIZATION_H