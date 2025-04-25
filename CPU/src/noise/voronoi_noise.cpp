// CPU/src/noise/voronoi_noise.cpp - minimal stub implementation
#include "../../include/noise/voronoi_noise.h"
#include <cmath>
#include <iostream>

// Minimal implementation that just returns a constant value
float voronoiNoise(float x, float y, int seed) {
    float cellSize = 1.0f;
    
    float nx = x / cellSize;
    float ny = y / cellSize;
    
    int xi = (int)floorf(nx);
    int yi = (int)floorf(ny);
    
    float minDist = 1000.0f;
    float secondMinDist = 1000.0f;
    
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int cx = xi + i;
            int cy = yi + j;
            
            unsigned int h = seed;
            h = (h * 747796405u) + (unsigned int)cx;
            h = (h * 747796405u) + (unsigned int)cy;
            h = (h ^ (h >> 12)) * 1664525u;
            h = (h ^ (h >> 19)) * 1013904223u;
            float fx = (h & 0xFFFF) / 65536.0f;
            
            h = (h * 747796405u) + (unsigned int)cx;
            h = (h * 747796405u) + (unsigned int)cy;
            h = (h ^ (h >> 12)) * 1664525u;
            h = (h ^ (h >> 19)) * 1013904223u;
            float fy = (h & 0xFFFF) / 65536.0f;
            
            float px = cx + fx;
            float py = cy + fy;
            
            float dx = px - nx;
            float dy = py - ny;
            float dist = sqrtf((dx*dx) + (dy*dy));
            
            if (dist < minDist) {
                secondMinDist = minDist;
                minDist = dist;
            } else if (dist < secondMinDist) {
                secondMinDist = dist;
            }
        }
    }
    
    return secondMinDist - minDist;
}