#include "../../include/terrain/terrain_generator.h"
#include "../../include/terrain/terrain_types.h"
#include "../../include/terrain/terrain_smoothing.h"
#include "../../include/noise/perlin_noise.h"
#include "../../include/noise/voronoi_noise.h"
#include "../../include/noise/noise_utils.h"
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>

// CPU version of generateTerrain kernel - matches GPU implementation
void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    // Process each point in the grid sequentially
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Match GPU bounds checking
            if (x < width && y < height) {
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
                
                // Create a more natural biome selector with more variation and less pattern
                float biomeSelector = (voronoiValue * 1.5f + elevation * 0.5f + moisture * 0.5f + localVar * 0.3f);
                biomeSelector = biomeSelector * (0.8f + 0.4f * noise(nx * 7.9f, ny * 11.3f, 0.0f));  

                float randVal = noise(nx * 23.4f, ny * 19.7f, 0.7f);
                
                // Variable region frequency to break up the pattern
                float regionFreq = 7.0f + noise(nx * 0.1f, ny * 0.1f, 0.0f) * 3.0f;
                int regionType = (int)(biomeSelector * regionFreq) % 6;

                const float waterChance = 0.55f;

                if (regionType == 5) {  // biome 5 = water
                    float chance = 0.5f + 0.5f * noise(nx * 123.45f, ny * 543.21f, 7.89f);
                    if (chance >= waterChance) {
                        // Replace with non-water biome (0–4 only)
                        regionType = (regionType + 1 + (int)(chance * 5)) % 5;
                    }
                }
                
                int terrainType;
                float desertVar = localVar + 0.1f * noise(nx * 13.7f, ny * 17.3f, 0.0f); // Add micro-variation

                // Assign terrain based on region type with local variations
                switch(regionType) {
                    case 0: // Desert regions
                        if (desertVar < 0.3f) terrainType = DESERT;
                        else if (desertVar < 0.6f) terrainType = SAND;
                        else terrainType = DUNE;
                        break;                
                    case 1: // Forest regions
                        if (localVar < 0.25f) terrainType = FOREST;
                        else if (localVar < 0.5f) terrainType = GRASS;
                        else if (localVar < 0.75f) terrainType = JUNGLE;
                        else terrainType = TAIGA;
                        break;
                        
                    case 2: // Mountain regions
                        if (elevation > 0.7f) {
                            if (localVar < 0.5f) terrainType = MOUNTAIN;
                            else terrainType = CLIFF;
                        } else {
                            if (localVar < 0.5f) terrainType = ROCK;
                            else terrainType = PLATEAU;
                        }
                        break;
                        
                    case 3: // Tundra regions
                        if (localVar < 0.3f) terrainType = TUNDRA;
                        else if (localVar < 0.6f) terrainType = SNOW;
                        else terrainType = GLACIER;
                        break;
                        
                    case 4: // Grassland regions
                        if (localVar < 0.25f) terrainType = GRASS;
                        else if (localVar < 0.5f) terrainType = PRAIRIE;
                        else if (localVar < 0.6f) terrainType = STEPPE;
                        else terrainType = SAVANNA;
                        break;
                        
                    case 5: // Water regions
                        if (localVar < 0.15f) terrainType = WATER;
                        else if (localVar < 0.3f) terrainType = BAY;
                        else if (localVar < 0.5f) terrainType = FJORD;
                        else terrainType = COVE;
                        break;
                        
                    default:
                        terrainType = GRASS; // Fallback
                }
                
                // Override for extreme elevations regardless of region
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
                
                terrain[y * width + x] = terrainType;
            }
        }
    }
}

void createPerlinNoiseTerrain(int* terrain, int width, int height,
    float scale, float offsetX, float offsetY) {
    
    // Print the same messages as the GPU version would
    printf("Generating terrain...\n");
    
    float adjustedScale = fmaxf(0.05f, fminf(fabs(scale), 100.0f));
    
    // Generate the initial terrain (no kernel launch syntax)
    generateTerrain(terrain, width, height, adjustedScale, offsetX, offsetY);

    // Allocate temporary terrain buffer (regular CPU allocation)
    int* tempTerrain = (int*)malloc(width * height * sizeof(int));
    if (!tempTerrain) {
        printf("Failed to allocate temporary terrain buffer\n");
        return;
    }

    // Step 1: Remove isolated noise
    printf("Removing isolated noise...\n");
    for (int i = 0; i < 5; i++) {
        removeIsolatedNoise(terrain, tempTerrain, width, height);
        memcpy(terrain, tempTerrain, width * height * sizeof(int));
    }

    // Step 2: Initial smoothing
    printf("Applying initial smoothing...\n");
    improvedSmoothTerrain(terrain, tempTerrain, width, height);
    memcpy(terrain, tempTerrain, width * height * sizeof(int));

    // Step 3: Remove vertical striping
    printf("Removing vertical striping...\n");
    removeVerticalStripes(terrain, tempTerrain, width, height);
    memcpy(terrain, tempTerrain, width * height * sizeof(int));

    // Step 4: Remove small patches
    printf("Cleaning up small patches...\n");
    cleanupSmallPatches(terrain, tempTerrain, width, height, 20);
    memcpy(terrain, tempTerrain, width * height * sizeof(int));

    // Step 5: Final multi-pass smoothing
    printf("Applying final smoothing passes...\n");
    const int smoothingPasses = 3;
    for (int i = 0; i < smoothingPasses; i++) {
        smoothTerrain(terrain, tempTerrain, width, height);
        memcpy(terrain, tempTerrain, width * height * sizeof(int));
    }

    // Clean up
    free(tempTerrain);
    printf("Terrain generation complete\n");
}