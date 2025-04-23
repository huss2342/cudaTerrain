// CPU/src/main.cpp - minimal version
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>

#include "../include/terrain/terrain_types.h"
#include "../include/terrain/terrain_generator.h"
#include "../include/visualization/visualization.h"

int main(int argc, char** argv) {
    try {
        // Add timing variables at the beginning
        clock_t start_time, end_time;
        double cpu_time_used;
        
        // Start the timer
        start_time = clock();
        
        std::cout << "Initializing terrain types..." << std::endl;
        TerrainTypes::initializeTerrainTypes();
        
        // Use very small dimensions for debugging
        int width = 16;
        int height = 16;
        
        if (argc > 1) {
            width = atoi(argv[1]);
            height = atoi(argv[1]);
        }
        
        std::cout << "Using dimensions: " << width << "x" << height << std::endl;
        
        // Set a specific seed for reproducibility
        int seed = 1234567;
        srand(seed);
        
        std::cout << "Using seed: " << seed << std::endl;
        
        // Allocate memory
        std::cout << "Allocating memory..." << std::endl;
        int* terrain = new int[width * height];
        if (!terrain) {
            std::cerr << "Failed to allocate terrain memory!" << std::endl;
            return 1;
        }
        
        unsigned char* image = new unsigned char[width * height * 3];
        if (!image) {
            std::cerr << "Failed to allocate image memory!" << std::endl;
            delete[] terrain;
            return 1;
        }
        
        // Generate terrain (simple version)
        std::cout << "Generating terrain..." << std::endl;
        generateTerrain(terrain, width, height, 1.0f, 0.0f, 0.0f);
        
        // Visualize terrain
        std::cout << "Visualizing terrain..." << std::endl;
        visualizeTerrain(terrain, image, width, height);
        
        // Save the image
        std::cout << "Saving image..." << std::endl;
        saveToPPM("minimal_terrain.ppm", image, width, height);
        
        // Clean up
        std::cout << "Cleaning up..." << std::endl;
        delete[] terrain;
        delete[] image;
        
        // Calculate and print execution time
        end_time = clock();
        cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Total execution time: %.2f seconds\n", cpu_time_used);
        
        std::cout << "Program completed successfully." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception in main!" << std::endl;
        return 1;
    }
}