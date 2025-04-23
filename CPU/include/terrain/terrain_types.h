#ifndef TERRAIN_TYPES_H
#define TERRAIN_TYPES_H

struct TerrainType {
    int id;
    const char* name;
    struct { int r, g, b; } color;
    bool walkable;
};

enum TerrainId {
    WATER, SAND, GRASS, ROCK, SNOW, LAVA, ICE, MUD, FOREST, DESERT,
    MOUNTAIN, SWAMP, JUNGLE, TUNDRA, SAVANNA, TAIGA, STEPPE, PRAIRIE,
    PLATEAU, CANYON, BADLANDS, MESA, OASIS, VOLCANO, GLACIER, FJORD,
    BAY, COVE, BEACH, CLIFF, DUNE
};

namespace TerrainTypes {
    // Declare the terrain types array
    extern TerrainType TERRAINS[31];
    
    // Function to initialize the terrain types
    void initializeTerrainTypes();
    
    // Function to get terrain by id
    const TerrainType* getTerrainById(int id);
}

#endif // TERRAIN_TYPES_H