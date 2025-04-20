#ifndef VORONOI_NOISE_H
#define VORONOI_NOISE_H

#include <cuda_runtime.h>

// Voronoi noise implementation
__device__ float voronoiNoise(float x, float y, int seed);

#endif // VORONOI_NOISE_H