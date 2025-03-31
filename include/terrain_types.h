#ifndef TERRAIN_TYPES_H
#define TERRAIN_TYPES_H

#include <cuda_runtime.h>

struct TerrainType {
    int id;
    const char* name;
    struct { int r, g, b; } color;
};

enum TerrainId {
    WATER, SAND, GRASS, ROCK, SNOW, LAVA, ICE, MUD, FOREST, DESERT,
    MOUNTAIN, SWAMP, JUNGLE, TUNDRA, SAVANNA, TAIGA, STEPPE, PRAIRIE,
    PLATEAU, CANYON, BADLANDS, MESA, OASIS, VOLCANO, GLACIER, FJORD,
    BAY, COVE, BEACH, CLIFF, DUNE
};

namespace TerrainTypes {
    // Declare as extern in the header
    extern __constant__ TerrainType TERRAINS[31];
    
    // Function to initialize the constant memory
    void initializeTerrainTypes();
    
    // Device and host function to get terrain by id
    __device__ __host__ const TerrainType* getTerrainById(int id);
}

#endif // TERRAIN_TYPES_H