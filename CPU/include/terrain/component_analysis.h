#ifndef COMPONENT_ANALYSIS_H
#define COMPONENT_ANALYSIS_H

// Connected component analysis
void identifyConnectedComponents(int* terrain, int* labels, int width, int height);
void propagateLabels(int* terrain, int* labels, int width, int height, bool* changed);
void removeSmallComponents(int* terrain, int* labels, int* output, int* componentSizes, int minSize, int width, int height);

#endif // COMPONENT_ANALYSIS_H