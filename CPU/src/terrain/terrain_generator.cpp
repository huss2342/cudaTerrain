// CPU/src/terrain/terrain_generator.cpp - absolute minimal version
#include "../../include/terrain/terrain_generator.h"
#include "../../include/terrain/terrain_types.h"
#include "../../include/terrain/terrain_smoothing.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/voronoi_noise.h"
#include "../../include/noise/noise_utils.h"
#include <cmath>
#include <cstring>
#include <new>
#include <stdexcept>
#include <iostream>

// CPU version of the GPU kernel - single pixel generation
void generateTerrainPixel(int* terrain, int width, int height, float scale, float offsetX, float offsetY, int x, int y) {
    if (!terrain) {
        std::cerr << "Null terrain pointer in generateTerrainPixel" << std::endl;
        throw std::runtime_error("Null terrain pointer");
    }

    if (x < 0 || x >= width || y < 0 || y >= height) {
        std::cerr << "Invalid coordinates in generateTerrainPixel: x=" << x << ", y=" << y << std::endl;
        throw std::runtime_error("Invalid coordinates");
    }

    long long idx = (long long)y * width + x;
    if (idx < 0 || idx >= (long long)width * height) {
        std::cerr << "Index out of bounds in generateTerrainPixel: idx=" << idx << std::endl;
        throw std::runtime_error("Index out of bounds");
    }

    // Coordinates for various noise functions
    float nx = (float)x / width * scale + offsetX;
    float ny = (float)y / height * scale + offsetY;
    
    // Domain warping - distort the space to break up the pattern
    float warpStrength = 0.2f;
    float warpX = nx + warpStrength * distributedNoise(nx * 2.0f, ny * 2.0f, 4, 0.5f, scale);
    float warpY = ny + warpStrength * distributedNoise(nx * 2.0f + 100.0f, ny * 2.0f + 100.0f, 4, 0.5f, scale);
    
    // Use warped coordinates for noise generation
    float noise1 = distributedNoise(warpX, warpY, 6, 0.5f, scale);
    float noise2 = distributedNoise(warpX * 2.3f, warpY * 1.7f, 4, 0.5f, scale);
    float noise3 = distributedNoise(warpX * 5.1f, warpY * 4.7f, 3, 0.5f, scale);
    float noise4 = distributedNoise(warpX * 11.3f, warpY * 7.9f, 2, 0.5f, scale);
    
    // Combine noise layers
    float elevation = 0.5f * noise1 + 0.25f * noise2 + 0.125f * noise3 + 0.125f * noise4;
    float moisture = distributedNoise(warpX + 100.0f, warpY + 100.0f, 4, 0.5f, scale);
    
    // Variable scale for Voronoi to create different sized regions
    float variableScale = scale * (0.8f + 0.4f * noise(nx * 0.01f, ny * 0.01f));
    float cellScale = 0.05f * variableScale;
    
    // Apply domain warping to Voronoi coordinates too
    float vwarpX = (float)x / width * cellScale + offsetX * 0.1f + 0.3f * noise(nx * 0.5f, ny * 0.5f);
    float vwarpY = (float)y / height * cellScale + offsetY * 0.1f + 0.3f * noise(nx * 0.5f + 50.0f, ny * 0.5f + 50.0f);
    
    // Multiple Voronoi layers with different frequencies
    float v1 = voronoiNoise(vwarpX, vwarpY);
    float v2 = voronoiNoise(vwarpX * 2.3f, vwarpY * 2.3f);
    float v3 = voronoiNoise(vwarpX * 0.5f, vwarpY * 0.5f);
    float voronoiValue = v1 * 0.5f + v2 * 0.3f + v3 * 0.2f;
    
    // Get local variation with domain warping
    float localScale = scale * 0.5f;
    float lwarpX = (float)x / width * localScale + offsetX + 500.0f + 0.2f * noise(nx * 1.5f, ny * 1.5f);
    float lwarpY = (float)y / height * localScale + offsetY + 500.0f + 0.2f * noise(nx * 1.5f + 200.0f, ny * 1.5f + 200.0f);
    float localVar = distributedNoise(lwarpX, lwarpY, 3, 0.5f, scale);
    
    // Create a more natural biome selector
    float biomeSelector = (voronoiValue * 1.5f + elevation * 0.5f + moisture * 0.5f + localVar * 0.3f);
    biomeSelector = biomeSelector * (0.8f + 0.4f * noise(nx * 7.9f, ny * 11.3f));  

    float randVal = noise(nx * 23.4f, ny * 19.7f);
    
    // Variable region frequency
    float regionFreq = 7.0f + noise(nx * 0.1f, ny * 0.1f) * 3.0f;
    int regionType = (int)(biomeSelector * regionFreq) % 6;

    const float waterChance = 0.55f;

    if (regionType == 5) {  // biome 5 = water
        float chance = 0.5f + 0.5f * noise(nx * 123.45f, ny * 543.21f);
        if (chance >= waterChance) {
            regionType = (regionType + 1 + (int)(chance * 5)) % 5;
        }
    }
    
    int terrainType;
    float desertVar = localVar + 0.1f * noise(nx * 13.7f, ny * 17.3f);

    // Assign terrain based on region type
    switch(regionType) {
        case 0:
            if (desertVar < 0.3f) terrainType = DESERT;
            else if (desertVar < 0.6f) terrainType = SAND;
            else terrainType = DUNE;
            break;                
        case 1:
            if (localVar < 0.25f) terrainType = FOREST;
            else if (localVar < 0.5f) terrainType = GRASS;
            else if (localVar < 0.75f) terrainType = JUNGLE;
            else terrainType = TAIGA;
            break;
        case 2:
            if (elevation > 0.7f) {
                if (localVar < 0.5f) terrainType = MOUNTAIN;
                else terrainType = CLIFF;
            } else {
                if (localVar < 0.5f) terrainType = ROCK;
                else terrainType = PLATEAU;
            }
            break;
        case 3:
            if (localVar < 0.3f) terrainType = TUNDRA;
            else if (localVar < 0.6f) terrainType = SNOW;
            else terrainType = GLACIER;
            break;
        case 4:
            if (localVar < 0.25f) terrainType = GRASS;
            else if (localVar < 0.5f) terrainType = PRAIRIE;
            else if (localVar < 0.6f) terrainType = STEPPE;
            else terrainType = SAVANNA;
            break;
        case 5:
            if (localVar < 0.15f) terrainType = WATER;
            else if (localVar < 0.3f) terrainType = BAY;
            else if (localVar < 0.5f) terrainType = FJORD;
            else terrainType = COVE;
            break;
        default:
            terrainType = GRASS;
    }
    
    // Override for extreme elevations
    if (elevation > 0.9f) {
        if (localVar < 0.3f) terrainType = MOUNTAIN;
        else if (localVar < 0.6f) terrainType = SNOW;
        else terrainType = GLACIER;
    }
    else if (elevation < 0.1f) {
        if (localVar < 0.3f) terrainType = WATER;
        else if (localVar < 0.6f) terrainType = BAY;
        else terrainType = COVE;
    }
    
    terrain[idx] = terrainType;
}

// Single-threaded terrain generation
void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    if (!terrain) {
        std::cerr << "Null terrain pointer in generateTerrain" << std::endl;
        throw std::runtime_error("Null terrain pointer");
    }

    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid dimensions in generateTerrain: width=" << width << ", height=" << height << std::endl;
        throw std::runtime_error("Invalid dimensions");
    }

    std::cout << "Starting terrain generation..." << std::endl;
    std::cout << "Parameters: width=" << width << ", height=" << height << ", scale=" << scale << std::endl;
    std::cout << "Offsets: X=" << offsetX << ", Y=" << offsetY << std::endl;

    try {
        for (int y = 0; y < height; y++) {
            if (y % 100 == 0) {
                std::cout << "Processing row " << y << " of " << height << std::endl;
            }
            for (int x = 0; x < width; x++) {
                generateTerrainPixel(terrain, width, height, scale, offsetX, offsetY, x, y);
            }
        }
        std::cout << "Terrain generation completed successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during terrain generation: " << e.what() << std::endl;
        throw;
    }
}

void createPerlinNoiseTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    std::cout << "Starting createPerlinNoiseTerrain..." << std::endl;
    float adjustedScale = fmaxf(0.05f, fminf(fabs(scale), 100.0f));
    std::cout << "Adjusted scale: " << adjustedScale << std::endl;
    
    try {
        // Allocate temporary buffer
        std::cout << "Allocating temporary buffer..." << std::endl;
        int* tempTerrain = new int[width * height];
        
        // Generate initial terrain
        std::cout << "Generating initial terrain..." << std::endl;
        generateTerrain(terrain, width, height, adjustedScale, offsetX, offsetY);

        std::cout << "Starting smoothing passes..." << std::endl;
        // Smoothing passes
        for (int i = 0; i < 5; i++) {
            std::cout << "Smoothing pass " << (i+1) << " of 5..." << std::endl;
            removeIsolatedNoise(terrain, tempTerrain, width, height);
            std::memcpy(terrain, tempTerrain, (long long)width * height * sizeof(int));
        }

        std::cout << "Applying improved smoothing..." << std::endl;
        improvedSmoothTerrain(terrain, tempTerrain, width, height);
        std::memcpy(terrain, tempTerrain, (long long)width * height * sizeof(int));

        std::cout << "Removing vertical stripes..." << std::endl;
        removeVerticalStripes(terrain, tempTerrain, width, height);
        std::memcpy(terrain, tempTerrain, (long long)width * height * sizeof(int));

        std::cout << "Cleaning up small patches..." << std::endl;
        cleanupSmallPatches(terrain, tempTerrain, width, height, 20);
        std::memcpy(terrain, tempTerrain, (long long)width * height * sizeof(int));

        std::cout << "Final smoothing passes..." << std::endl;
        for (int i = 0; i < 3; i++) {
            std::cout << "Final smoothing pass " << (i+1) << " of 3..." << std::endl;
            smoothTerrain(terrain, tempTerrain, width, height);
            std::memcpy(terrain, tempTerrain, (long long)width * height * sizeof(int));
        }

        delete[] tempTerrain;
        std::cout << "createPerlinNoiseTerrain completed successfully" << std::endl;
    }
    catch (const std::bad_alloc&) {
        std::cerr << "Failed to allocate memory for terrain generation" << std::endl;
        throw std::runtime_error("Failed to allocate memory for terrain generation");
    }
    catch (const std::exception& e) {
        std::cerr << "Error in createPerlinNoiseTerrain: " << e.what() << std::endl;
        throw;
    }
}