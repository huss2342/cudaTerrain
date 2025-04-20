#include "../../include/noise/voronoi_noise.h"
#include <math.h>

// Simple 2D Voronoi noise implementation
__device__ float voronoiNoise(float x, float y, int seed) {
    float cellSize = 1.0f;
    
    // Normalize coordinates
    float nx = x / cellSize;
    float ny = y / cellSize;
    
    // Get integer cell coordinates
    int xi = floorf(nx);
    int yi = floorf(ny);
    
    float minDist = 1000.0f;
    float secondMinDist = 1000.0f;
    
    // Check surrounding cells
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int cx = xi + i;
            int cy = yi + j;
            
            // Get random position in cell using improved hash
            unsigned int h = seed;
            h = (h * 747796405) + cx;
            h = (h * 747796405) + cy;
            h = (h ^ (h >> 12)) * 1664525;
            h = (h ^ (h >> 19)) * 1013904223;
            float fx = (h & 0xFFFF) / 65536.0f;
            
            h = (h * 747796405) + cx;
            h = (h * 747796405) + cy;
            h = (h ^ (h >> 12)) * 1664525;
            h = (h ^ (h >> 19)) * 1013904223;
            float fy = (h & 0xFFFF) / 65536.0f;
            
            // Feature point position (in normalized space)
            float px = cx + fx;
            float py = cy + fy;
            
            // Distance to feature point
            float dx = px - nx;
            float dy = py - ny;
            float dist = sqrtf((dx*dx) + (dy*dy));
            
            // Update distances
            if (dist < minDist) {
                secondMinDist = minDist;
                minDist = dist;
            } else if (dist < secondMinDist) {
                secondMinDist = dist;
            }
        }
    }
    
    // Return difference between distances as cell boundary
    return secondMinDist - minDist;
}