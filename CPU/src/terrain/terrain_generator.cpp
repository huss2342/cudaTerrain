// CPU/src/terrain/terrain_generator.cpp - absolute minimal version
#include "../../include/terrain/terrain_generator.h"
#include "../../include/terrain/terrain_types.h"
#include <iostream>

// Minimal implementation that doesn't use any noise functions
void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    try {
        std::cout << "Starting minimal terrain generation without noise functions..." << std::endl;
        
        // Validate input parameters
        if (!terrain) {
            std::cerr << "ERROR: Null terrain pointer!" << std::endl;
            return;
        }
        
        if (width <= 0 || height <= 0) {
            std::cerr << "ERROR: Invalid terrain dimensions: " << width << "x" << height << std::endl;
            return;
        }
        
        // Just fill the terrain with a checkerboard pattern
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                
                // Simple checkerboard pattern with 5 terrain types
                int terrainType = ((x / 4) + (y / 4)) % 5;
                
                // Ensure terrain type is valid
                if (terrainType < 0 || terrainType > 4) {
                    terrainType = 0;
                }
                
                terrain[idx] = terrainType;
            }
        }
        
        std::cout << "Terrain generation completed successfully." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during terrain generation: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception during terrain generation!" << std::endl;
    }
}

void createPerlinNoiseTerrain(int* terrain, int width, int height,
    float scale, float offsetX, float offsetY) {
    
    try {
        // Just call the simplified terrain generator
        std::cout << "Bypassing noise algorithms completely..." << std::endl;
        generateTerrain(terrain, width, height, scale, offsetX, offsetY);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in createPerlinNoiseTerrain: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception in createPerlinNoiseTerrain!" << std::endl;
    }
}