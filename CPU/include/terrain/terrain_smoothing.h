#ifndef TERRAIN_SMOOTHING_H
#define TERRAIN_SMOOTHING_H

// Smoothing operations
void smoothTerrain(int* terrain, int* output, int width, int height);
void removeVerticalStripes(int* terrain, int* output, int width, int height);
void cleanupSmallPatches(int* terrain, int* output, int width, int height, int minRegionSize);
void improvedSmoothTerrain(int* terrain, int* output, int width, int height);
void removeIsolatedNoise(int* terrain, int* output, int width, int height);

#endif // TERRAIN_SMOOTHING_H