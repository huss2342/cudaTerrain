// CPU/src/noise/voronoi_noise.cpp - minimal stub implementation
#include "../../include/noise/voronoi_noise.h"
#include "../../include/noise/perlin_noise.h"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// Hash function for Voronoi noise
int hash(int x, int y) {
    int h = x * 374761393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return h ^ (h >> 16);
}

float voronoiNoise(float x, float y) {
    float cellX = floor(x);
    float cellY = floor(y);
    float minDist = 1.0f;

    for (int offsetY = -1; offsetY <= 1; offsetY++) {
        for (int offsetX = -1; offsetX <= 1; offsetX++) {
            float neighborX = cellX + offsetX;
            float neighborY = cellY + offsetY;
            
            // Generate a random point within the cell
            float pointX = neighborX + noise(neighborX, neighborY);
            float pointY = neighborY + noise(neighborY, neighborX);
            
            // Calculate distance to the point
            float dx = pointX - x;
            float dy = pointY - y;
            float dist = sqrt(dx * dx + dy * dy);
            
            minDist = std::min(minDist, dist);
        }
    }

    return minDist;
}

float hash(int n) {
    try {
        // Ensure n is within a reasonable range to prevent overflow
        n = n & 0x7fffffff;  // Keep only lower 31 bits
        
        n = (n << 13) ^ n;
        n = (n * (n * n * 15731 + 789221) + 1376312589);
        
        float result = 1.0f - (static_cast<float>(n & 0x7fffffff) / 1073741824.0f);
        
        // Check for numerical issues
        if (std::isnan(result) || std::isinf(result)) {
            throw std::runtime_error("Invalid hash calculation");
        }
        
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in hash function: " << e.what() << std::endl;
        return 0.5f;  // Return middle value on error
    }
}