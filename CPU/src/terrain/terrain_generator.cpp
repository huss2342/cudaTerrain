// CPU/src/terrain/terrain_generator.cpp - absolute minimal version
#include "../../include/terrain/terrain_generator.h"
#include "../../include/terrain/terrain_types.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/voronoi_noise.h"
#include "../../include/noise/noise_utils.h"
#include <thread>
#include <vector>
#include <iostream>
#include <cmath>

// Helper function for multi-threaded terrain generation
void generateTerrainChunk(int* terrain, int width, int height, float scale, 
                         float offsetX, float offsetY, int startY, int endY) {
    try {
        for (int y = startY; y < endY; y++) {
            for (int x = 0; x < width; x++) {
                if (y >= height || x >= width) {
                    std::cerr << "Error: Attempting to access out of bounds: x=" << x << ", y=" << y << std::endl;
                    continue;
                }

                // Coordinates for various noise functions
                float nx = (float)x / width * scale + offsetX;
                float ny = (float)y / height * scale + offsetY;
                
                // Domain warping - distort the space to break up the pattern
                float warpStrength = 0.2f;
                float warpX = nx + warpStrength * distributedNoise(nx * 2.0f, ny * 2.0f, 0.0f, 4);
                float warpY = ny + warpStrength * distributedNoise(nx * 2.0f + 100.0f, ny * 2.0f + 100.0f, 0.0f, 4);
                
                // Use warped coordinates for noise generation
                float noise1 = distributedNoise(warpX, warpY, 0.0f, 6);
                float noise2 = distributedNoise(warpX * 2.3f, warpY * 1.7f, 0.5f, 4);
                float noise3 = distributedNoise(warpX * 5.1f, warpY * 4.7f, 1.0f, 3);
                float noise4 = distributedNoise(warpX * 11.3f, warpY * 7.9f, 2.0f, 2);
                
                // Combine noise layers
                float elevation = 0.5f * noise1 + 0.25f * noise2 + 0.125f * noise3 + 0.125f * noise4;
                float moisture = distributedNoise(warpX + 100.0f, warpY + 100.0f, 1.0f, 4);
                
                // Variable scale for Voronoi to create different sized regions
                float variableScale = scale * (0.8f + 0.4f * noise(nx * 0.01f, ny * 0.01f, 0.5f));
                float cellScale = 0.05f * variableScale;
                
                // Apply domain warping to Voronoi coordinates too
                float vwarpX = (float)x / width * cellScale + offsetX * 0.1f + 0.3f * noise(nx * 0.5f, ny * 0.5f, 0.0f);
                float vwarpY = (float)y / height * cellScale + offsetY * 0.1f + 0.3f * noise(nx * 0.5f + 50.0f, ny * 0.5f + 50.0f, 0.0f);
                
                // Multiple Voronoi layers with different frequencies
                float v1 = voronoiNoise(vwarpX, vwarpY, 12345);
                float v2 = voronoiNoise(vwarpX * 2.3f, vwarpY * 2.3f, 54321);
                float v3 = voronoiNoise(vwarpX * 0.5f, vwarpY * 0.5f, 98765);
                float voronoiValue = v1 * 0.5f + v2 * 0.3f + v3 * 0.2f;
                
                // Get local variation with domain warping
                float localScale = scale * 0.5f;
                float lwarpX = (float)x / width * localScale + offsetX + 500.0f + 0.2f * noise(nx * 1.5f, ny * 1.5f, 0.0f);
                float lwarpY = (float)y / height * localScale + offsetY + 500.0f + 0.2f * noise(nx * 1.5f + 200.0f, ny * 1.5f + 200.0f, 0.0f);
                float localVar = distributedNoise(lwarpX, lwarpY, 2.0f, 3);
                
                // Create a more natural biome selector
                float biomeSelector = (voronoiValue * 1.5f + elevation * 0.5f + moisture * 0.5f + localVar * 0.3f);
                biomeSelector = biomeSelector * (0.8f + 0.4f * noise(nx * 7.9f, ny * 11.3f, 0.0f));
                
                // Determine terrain type based on combined factors
                int terrainType;
                float normalizedSelector = biomeSelector * 0.5f + 0.5f; // Normalize to 0-1 range
                
                if (elevation > 0.8f) {
                    if (moisture > 0.6f) terrainType = 24; // GLACIER
                    else if (moisture > 0.3f) terrainType = 4; // SNOW
                    else terrainType = 10; // MOUNTAIN
                }
                else if (elevation > 0.6f) {
                    if (moisture > 0.7f) terrainType = 25; // FJORD
                    else if (moisture > 0.4f) terrainType = 19; // CANYON
                    else terrainType = 21; // MESA
                }
                else if (elevation > 0.4f) {
                    if (moisture > 0.6f) terrainType = 8; // FOREST
                    else if (moisture > 0.3f) terrainType = 15; // TAIGA
                    else terrainType = 18; // PLATEAU
                }
                else if (elevation > 0.2f) {
                    if (moisture > 0.6f) terrainType = 12; // JUNGLE
                    else if (moisture > 0.3f) terrainType = 2; // GRASS
                    else terrainType = 9; // DESERT
                }
                else {
                    if (moisture > 0.7f) terrainType = 11; // SWAMP
                    else if (moisture > 0.4f) terrainType = 0; // WATER
                    else terrainType = 1; // SAND
                }
                
                // Override for extreme elevations regardless of region
                if (elevation > 0.9f) {
                    if (localVar < 0.3f) terrainType = 10; // MOUNTAIN
                    else if (localVar < 0.6f) terrainType = 4; // SNOW
                    else terrainType = 24; // GLACIER
                }
                else if (elevation < 0.1f) {
                    if (localVar < 0.3f) terrainType = 0; // WATER
                    else if (localVar < 0.6f) terrainType = 26; // BAY
                    else terrainType = 27; // COVE
                }
                
                // Add bounds checking before writing to terrain array
                size_t idx = (size_t)y * width + x;
                if (idx >= (size_t)width * height) {
                    std::cerr << "Error: Array index out of bounds: " << idx << std::endl;
                    continue;
                }
                terrain[idx] = terrainType;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in generateTerrainChunk: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception in generateTerrainChunk!" << std::endl;
    }
}

void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    std::cout << "Starting terrain generation..." << std::endl;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << "Processing pixel (" << x << "," << y << ")" << std::endl;
            
            // Coordinates for various noise functions
            float nx = (float)x / width * scale + offsetX;
            float ny = (float)y / height * scale + offsetY;
            
            // Domain warping - distort the space to break up the pattern
            float warpStrength = 0.2f;
            float warpX = nx + warpStrength * distributedNoise(nx * 2.0f, ny * 2.0f, 0.0f, 4);
            float warpY = ny + warpStrength * distributedNoise(nx * 2.0f + 100.0f, ny * 2.0f + 100.0f, 0.0f, 4);
            
            std::cout << "  Warping complete" << std::endl;
            
            // Use warped coordinates for noise generation
            float noise1 = distributedNoise(warpX, warpY, 0.0f, 6);
            float noise2 = distributedNoise(warpX * 2.3f, warpY * 1.7f, 0.5f, 4);
            float noise3 = distributedNoise(warpX * 5.1f, warpY * 4.7f, 1.0f, 3);
            float noise4 = distributedNoise(warpX * 11.3f, warpY * 7.9f, 2.0f, 2);
            
            std::cout << "  Noise layers generated" << std::endl;
            
            // Combine noise layers
            float elevation = 0.5f * noise1 + 0.25f * noise2 + 0.125f * noise3 + 0.125f * noise4;
            float moisture = distributedNoise(warpX + 100.0f, warpY + 100.0f, 1.0f, 4);
            
            // Variable scale for Voronoi to create different sized regions
            float variableScale = scale * (0.8f + 0.4f * noise(nx * 0.01f, ny * 0.01f, 0.5f));
            float cellScale = 0.05f * variableScale;
            
            // Apply domain warping to Voronoi coordinates too
            float vwarpX = (float)x / width * cellScale + offsetX * 0.1f + 0.3f * noise(nx * 0.5f, ny * 0.5f, 0.0f);
            float vwarpY = (float)y / height * cellScale + offsetY * 0.1f + 0.3f * noise(nx * 0.5f + 50.0f, ny * 0.5f + 50.0f, 0.0f);
            
            // Multiple Voronoi layers with different frequencies
            float v1 = voronoiNoise(vwarpX, vwarpY, 12345);
            float v2 = voronoiNoise(vwarpX * 2.3f, vwarpY * 2.3f, 54321);
            float v3 = voronoiNoise(vwarpX * 0.5f, vwarpY * 0.5f, 98765);
            float voronoiValue = v1 * 0.5f + v2 * 0.3f + v3 * 0.2f;
            
            // Get local variation with domain warping
            float localScale = scale * 0.5f;
            float lwarpX = (float)x / width * localScale + offsetX + 500.0f + 0.2f * noise(nx * 1.5f, ny * 1.5f, 0.0f);
            float lwarpY = (float)y / height * localScale + offsetY + 500.0f + 0.2f * noise(nx * 1.5f + 200.0f, ny * 1.5f + 200.0f, 0.0f);
            float localVar = distributedNoise(lwarpX, lwarpY, 2.0f, 3);
            
            // Create a more natural biome selector
            float biomeSelector = (voronoiValue * 1.5f + elevation * 0.5f + moisture * 0.5f + localVar * 0.3f);
            biomeSelector = biomeSelector * (0.8f + 0.4f * noise(nx * 7.9f, ny * 11.3f, 0.0f));
            
            // Determine terrain type based on combined factors
            int terrainType;
            
            if (elevation > 0.8f) {
                if (moisture > 0.6f) terrainType = 24; // GLACIER
                else if (moisture > 0.3f) terrainType = 4; // SNOW
                else terrainType = 10; // MOUNTAIN
            }
            else if (elevation > 0.6f) {
                if (moisture > 0.7f) terrainType = 25; // FJORD
                else if (moisture > 0.4f) terrainType = 19; // CANYON
                else terrainType = 21; // MESA
            }
            else if (elevation > 0.4f) {
                if (moisture > 0.6f) terrainType = 8; // FOREST
                else if (moisture > 0.3f) terrainType = 15; // TAIGA
                else terrainType = 18; // PLATEAU
            }
            else if (elevation > 0.2f) {
                if (moisture > 0.6f) terrainType = 12; // JUNGLE
                else if (moisture > 0.3f) terrainType = 2; // GRASS
                else terrainType = 9; // DESERT
            }
            else {
                if (moisture > 0.7f) terrainType = 11; // SWAMP
                else if (moisture > 0.4f) terrainType = 0; // WATER
                else terrainType = 1; // SAND
            }
            
            // Override for extreme elevations regardless of region
            if (elevation > 0.9f) {
                if (localVar < 0.3f) terrainType = 10; // MOUNTAIN
                else if (localVar < 0.6f) terrainType = 4; // SNOW
                else terrainType = 24; // GLACIER
            }
            else if (elevation < 0.1f) {
                if (localVar < 0.3f) terrainType = 0; // WATER
                else if (localVar < 0.6f) terrainType = 26; // BAY
                else terrainType = 27; // COVE
            }
            
            terrain[y * width + x] = terrainType;
        }
    }
    
    std::cout << "Terrain generation complete" << std::endl;
}

void createPerlinNoiseTerrain(int* terrain, int width, int height,
                             float scale, float offsetX, float offsetY) {
    float adjustedScale = std::max(0.05f, std::min(std::abs(scale), 100.0f));
    generateTerrain(terrain, width, height, adjustedScale, offsetX, offsetY);
}