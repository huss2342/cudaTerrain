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
    // Change the declaration to this
    extern __constant__ TerrainType TERRAINS[];
    __device__ __host__ inline const TerrainType* getTerrainById(int id);
}

#endif // TERRAIN_TYPES_H