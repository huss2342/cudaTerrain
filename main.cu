#include <stdio.h>

// CUDA kernel function to print "Hello World"
// __global__ void helloFromGPU()
// {
//     // Get the thread ID
//     int threadId = threadIdx.x;
//     printf("Hello World from GPU thread %d!\n", threadId);
// }

struct TerrainType {
    int id;
    const char* name;
    Color color;
};

// terrain types
namespace TerrainTypes {
    __constant__ TerrainType WATER    = {0, "Water", {0, 0, 255}};
    __constant__ TerrainType SAND     = {1, "Sand", {255, 255, 0}};
    __constant__ TerrainType GRASS    = {2, "Grass", {0, 255, 0}};
    __constant__ TerrainType ROCK     = {3, "Rock", {128, 128, 128}};
    __constant__ TerrainType SNOW     = {4, "Snow", {255, 255, 255}};
    __constant__ TerrainType LAVA     = {5, "Lava", {255, 0, 0}};
    __constant__ TerrainType ICE      = {6, "Ice", {0, 255, 255}};
    __constant__ TerrainType MUD      = {7, "Mud", {139, 69, 19}};
    __constant__ TerrainType FOREST   = {8, "Forest", {0, 100, 0}};
    __constant__ TerrainType DESERT   = {9, "Desert", {255, 165, 0}};
    __constant__ TerrainType MOUNTAIN = {10, "Mountain", {139, 69, 19}};
    __constant__ TerrainType SWAMP    = {11, "Swamp", {46, 139, 87}};
    __constant__ TerrainType JUNGLE   = {12, "Jungle", {34, 139, 34}};
    __constant__ TerrainType TUNDRA   = {13, "Tundra", {0, 128, 128}};
    __constant__ TerrainType SAVANNA  = {14, "Savanna", {218, 165, 32}};
    __constant__ TerrainType TAIGA    = {15, "Taiga", {0, 128, 0}};
    __constant__ TerrainType STEPPE   = {16, "Steppe", {210, 180, 140}};
    __constant__ TerrainType PRAIRIE  = {17, "Prairie", {255, 228, 196}};
    __constant__ TerrainType PLATEAU  = {18, "Plateau", {205, 133, 63}};
    __constant__ TerrainType CANYON   = {19, "Canyon", {139, 69, 19}};
    __constant__ TerrainType BADLANDS = {20, "Badlands", {139, 69, 19}};
    __constant__ TerrainType MESA     = {21, "Mesa", {139, 69, 19}};
    __constant__ TerrainType OASIS    = {22, "Oasis", {0, 255, 127}};
    __constant__ TerrainType VOLCANO  = {23, "Volcano", {255, 0, 0}};
    __constant__ TerrainType GLACIER  = {24, "Glacier", {0, 255, 255}};
    __constant__ TerrainType FJORD    = {25, "Fjord", {0, 0, 255}};
    __constant__ TerrainType BAY      = {26, "Bay", {0, 0, 255}};
    __constant__ TerrainType COVE     = {27, "Cove", {0, 0, 255}};
    __constant__ TerrainType BEACH    = {28, "Beach", {255, 255, 0}};
    __constant__ TerrainType CLIFF    = {29, "Cliff", {139, 69, 19}};
    __constant__ TerrainType DUNE     = {30, "Dune", {255, 255, 0}};
}


int main() {
    // Print from CPU
    printf("Hello World from CPU!\n");
    
    // Launch the kernel with 5 threads
    helloFromGPU<<<1, 5>>>();
    
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    
    return 0;
}

void renderTerrain(int* terrain, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int terrainType = terrain[y * width + x];
            Color color = getColorForTerrainType(terrainType);
            drawPixel(x, y, color);
        }
    }
}