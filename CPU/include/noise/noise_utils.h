#ifndef NOISE_UTILS_H
#define NOISE_UTILS_H

// Enhanced noise functions
float enhancedNoise(float x, float y, float scale);
float distributedNoise(float x, float y, int octaves, float persistence, float scale);
float combinedNoise(float x, float y, float scale);

#endif // NOISE_UTILS_H