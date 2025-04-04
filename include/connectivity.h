#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <utility> // for std::pair
#include <cmath>   // for math functions
#include <cstring> // for memcpy

// Main function to connect landmasses and expand walkable areas
void connectLandmasses(int* d_terrain, int width, int height);

// Function to expand existing walkable areas
void expandWalkableAreas(int* d_terrain, int width, int height);

#endif // CONNECTIVITY_H