#include <stdio.h>

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
    __device__ __constant__ const TerrainType TERRAINS[] = {
        {0,  "Water",    {0,   0,   255}},
        {1,  "Sand",     {255, 255, 0  }},
        {2,  "Grass",    {0,   255, 0  }},
        {3,  "Rock",     {128, 128, 128}},
        {4,  "Snow",     {255, 255, 255}},
        {5,  "Lava",     {255, 0,   0  }},
        {6,  "Ice",      {0,   255, 255}},
        {7,  "Mud",      {139, 69,  19 }},
        {8,  "Forest",   {0,   100, 0  }},
        {9,  "Desert",   {255, 165, 0  }},
        {10, "Mountain", {139, 69,  19 }},
        {11, "Swamp",    {46,  139, 87 }},
        {12, "Jungle",   {34,  139, 34 }},
        {13, "Tundra",   {0,   128, 128}},
        {14, "Savanna",  {218, 165, 32 }},
        {15, "Taiga",    {0,   128, 0  }},
        {16, "Steppe",   {210, 180, 140}},
        {17, "Prairie",  {255, 228, 196}},
        {18, "Plateau",  {205, 133, 63 }},
        {19, "Canyon",   {139, 69,  19 }},
        {20, "Badlands", {139, 69,  19 }},
        {21, "Mesa",     {139, 69,  19 }},
        {22, "Oasis",    {0,   255, 127}},
        {23, "Volcano",  {255, 0,   0  }},
        {24, "Glacier",  {0,   255, 255}},
        {25, "Fjord",    {0,   0,   255}},
        {26, "Bay",      {0,   0,   255}},
        {27, "Cove",     {0,   0,   255}},
        {28, "Beach",    {255, 255, 0  }},
        {29, "Cliff",    {139, 69,  19 }},
        {30, "Dune",     {255, 255, 0  }}
    };

    __device__ __host__ inline const TerrainType* getTerrainById(int id) {
        return (id >= 0 && id <= DUNE) ? &TERRAINS[id] : nullptr;
    }
}

int main() {
    printf("Hello World from CPU!\n");
    
    // const TerrainType* terrain = TerrainTypes::getTerrainById(GRASS);
    for (int i = 0; i <= DUNE; i++) {
        const TerrainType* terrain = TerrainTypes::getTerrainById(i);
        printf("terrain: %s (RGB: %d,%d,%d)\n", 
            terrain->name, terrain->color.r, terrain->color.g, terrain->color.b);
    }
    return 0;
}