// CPU/src/main.cpp - minimal version
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <cstring>

#include "../include/terrain/terrain_types.h"
#include "../include/noise/perlin_noise.h"
#include "../include/terrain/terrain_generator.h"
#include "../include/visualization/visualization.h"
#include "../include/terrain/terrain_height.h"

int main(int argc, char** argv) {
    try {
        std::cout << "1. Starting program" << std::endl;
        
        // Initialize parameters
        float scale = 80.0f;
        int size = 4;
        bool useHeight = true;
        
        std::cout << "2. Parameters initialized" << std::endl;
        
        // Generate seed and offsets
        int seed = 1234567;
        float randomOffsetX = (seed % 100) * 1.27f;
        float randomOffsetY = (seed % 100) * 2.53f;
        
        std::cout << "3. Seed and offsets generated" << std::endl;
        
        // Allocate memory
        int width = size;
        int height = size;
        int* terrain = new int[width * height];
        unsigned char* image = new unsigned char[width * height * 3];
        
        std::cout << "4. Memory allocated" << std::endl;
        
        // Initialize terrain types
        TerrainTypes::initializeTerrainTypes();
        
        std::cout << "5. Terrain types initialized" << std::endl;
        
        // Generate terrain
        createPerlinNoiseTerrain(terrain, width, height, scale, randomOffsetX, randomOffsetY);
        
        std::cout << "6. Terrain generated" << std::endl;
        
        // Visualize
        visualizeTerrain(terrain, image, width, height);
        
        std::cout << "7. Visualization complete" << std::endl;
        
        // Save
        saveToPPM("test_terrain.ppm", image, width, height);
        
        std::cout << "8. Image saved" << std::endl;
        
        // Cleanup
        delete[] terrain;
        delete[] image;
        
        std::cout << "9. Cleanup complete" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }
}