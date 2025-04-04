#include "../include/terrain_types.h"

// Define the constant array once
__constant__ TerrainType TerrainTypes::TERRAINS[31];

// Host-side array for initialization
static const TerrainType terrainData[31] = {
    {0,  "Water",    {0,   0,   255}}, 
    {1,  "Sand",     {255, 235, 130}}, 
    {2,  "Grass",    {0,   200, 0  }}, 
    {3,  "Rock",     {128, 128, 128}}, 
    {4,  "Snow",     {255, 255, 255}}, 
    {5,  "Lava",     {255, 0,   0  }}, 
    {6,  "Ice",      {0,   255, 255}}, 
    {7,  "Mud",      {139, 69,  19 }}, 
    {8,  "Forest",   {0,   100, 0  }}, 
    {9,  "Desert",   {255, 165, 0  }}, 
    {10, "Mountain", {105, 105, 105}}, 
    {11, "Swamp",    {46,  139, 87 }}, 
    {12, "Jungle",   {34,  139, 34 }}, 
    {13, "Tundra",   {0,   128, 128}}, 
    {14, "Savanna",  {218, 165, 32 }}, 
    {15, "Taiga",    {0,   128, 0  }}, 
    {16, "Steppe",   {210, 180, 140}}, 
    {17, "Prairie",  {255, 228, 196}}, 
    {18, "Plateau",  {205, 133, 63 }}, 
    {19, "Canyon",   {160, 82,  45 }}, 
    {20, "Badlands", {178, 34,  34 }}, 
    {21, "Mesa",     {222, 184, 135}}, 
    {22, "Oasis",    {0,   255, 127}}, 
    {23, "Volcano",  {128, 0,   0  }}, 
    {24, "Glacier",  {176, 224, 230}}, 
    {25, "Fjord",    {70,  130, 180}}, 
    {26, "Bay",      {0,   105, 148}}, 
    {27, "Cove",     {100, 149, 237}}, 
    {28, "Beach",    {238, 214, 175}}, 
    {29, "Cliff",    {112, 128, 144}}, 
    {30, "Dune",     {194, 178, 128}}  
};

// Function to initialize the constant memory
void TerrainTypes::initializeTerrainTypes() {
    cudaMemcpyToSymbol(TERRAINS, terrainData, sizeof(TerrainType) * 31);
}

// Implementation of getTerrainById
__device__ __host__ const TerrainType* TerrainTypes::getTerrainById(int id) {
    return (id >= 0 && id <= DUNE) ? &TERRAINS[id] : nullptr;
}