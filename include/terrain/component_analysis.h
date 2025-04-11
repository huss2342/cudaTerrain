#ifndef COMPONENT_ANALYSIS_H
#define COMPONENT_ANALYSIS_H

#include <cuda_runtime.h>

// Connected component analysis
__global__ void identifyConnectedComponents(int* terrain, int* labels, int width, int height);
__global__ void propagateLabels(int* terrain, int* labels, int width, int height, bool* changed);
__global__ void removeSmallComponents(int* terrain, int* labels, int* output, int* componentSizes, int minSize, int width, int height);

#endif // COMPONENT_ANALYSIS_H