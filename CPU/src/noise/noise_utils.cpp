// CPU/src/noise/noise_utils.cpp - minimal stub implementation
#include "../../include/noise/noise_utils.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/voronoi_noise.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

float distributedNoise(float x, float y, int octaves, float persistence, float scale) {
    float total = 0;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maxValue = 0;

    for (int i = 0; i < octaves; i++) {
        total += noise(x * frequency / scale, y * frequency / scale) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }

    return total / maxValue;
}

float enhancedNoise(float x, float y, float scale) {
    float perlinValue = distributedNoise(x, y, 6, 0.5f, scale);
    float voronoiValue = voronoiNoise(x / scale, y / scale);
    return (perlinValue + voronoiValue) / 2.0f;
}

float combinedNoise(float x, float y, float scale) {
    float perlin = enhancedNoise(x, y, scale);
    float voronoi = voronoiNoise(x / scale, y / scale);
    return (perlin * 0.7f + voronoi * 0.3f);
}