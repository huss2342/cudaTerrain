#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <limits>

#include "../include/terrain/terrain_types.h"
#include "../include/noise/perlin_noise.h"
#include "../include/terrain/terrain_generator.h"
#include "../include/visualization/visualization.h"
#include "../include/terrain/terrain_height.h"

int main(int argc, char** argv) {
    try {
        // Fixed parameters to match GPU version
        float scale = 80.0f;
        int size = 4096;
        int seed = 1234567;
        
        // Generate offsets from seed
        float randomOffsetX = (seed % 100) * 1.27f;
        float randomOffsetY = (seed % 100) * 2.53f;
        
        std::cout << "Generating terrain..." << std::endl;
        
        // Calculate array sizes
        int width = size;
        int height = size;
        long long totalSize = (long long)width * height;
        long long imageSize = totalSize * 3;
        
        // Check for potential overflow
        if (totalSize > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Size too large, would cause integer overflow");
        }
        
        std::cout << "Allocating memory..." << std::endl;
        
        // Allocate memory with checks
        int* terrain = nullptr;
        float* heightMap = nullptr;
        float* erodedHeightMap = nullptr;
        unsigned char* image = nullptr;
        
        try {
            terrain = new int[totalSize];
            heightMap = new float[totalSize];
            erodedHeightMap = new float[totalSize];
            image = new unsigned char[imageSize];
            
            if (!terrain || !heightMap || !erodedHeightMap || !image) {
                throw std::bad_alloc();
            }
            
            // Zero out the arrays
            std::memset(terrain, 0, totalSize * sizeof(int));
            std::memset(heightMap, 0, totalSize * sizeof(float));
            std::memset(erodedHeightMap, 0, totalSize * sizeof(float));
            std::memset(image, 0, imageSize * sizeof(unsigned char));
        }
        catch (const std::bad_alloc&) {
            delete[] terrain;
            delete[] heightMap;
            delete[] erodedHeightMap;
            delete[] image;
            throw std::runtime_error("Failed to allocate memory");
        }
        
        std::cout << "Initializing terrain types..." << std::endl;
        TerrainTypes::initializeTerrainTypes();
        
        std::cout << "Generating terrain data..." << std::endl;
        createPerlinNoiseTerrain(terrain, width, height, scale, randomOffsetX, randomOffsetY);
        
        std::cout << "Generating height map..." << std::endl;
        generateHeightMap(terrain, heightMap, width, height, scale, randomOffsetX, randomOffsetY);
        
        std::cout << "Applying erosion..." << std::endl;
        simulateErosion(heightMap, erodedHeightMap, width, height, 3, 0.15f);
        
        std::cout << "Visualizing terrain..." << std::endl;
        visualizeTerrainWithHeight(terrain, erodedHeightMap, image, width, height);
        
        // Save
        std::string filename = "terrain_4096x4096.ppm";
        saveToPPM(filename.c_str(), image, width, height);
        
        std::cout << "Terrain generation complete!" << std::endl;
        std::cout << "Output saved to: " << filename << std::endl;
        
        // Cleanup
        delete[] terrain;
        delete[] heightMap;
        delete[] erodedHeightMap;
        delete[] image;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return 1;
    }
}