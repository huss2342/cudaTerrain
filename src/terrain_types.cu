#include "../include/terrain_types.h"

// Define the constant array once
__constant__ TerrainType TerrainTypes::TERRAINS[31];

// Host-side array for initialization
static const TerrainType terrainData[31] = {
    {0,  "Water",    {0,   0,   255}, false}, 
    {1,  "Sand",     {255, 235, 130}, true}, 
    {2,  "Grass",    {0,   200, 0  }, true}, 
    {3,  "Rock",     {128, 128, 128}, true}, 
    {4,  "Snow",     {255, 255, 255}, true}, 
    {5,  "Lava",     {255, 0,   0  }, false}, 
    {6,  "Ice",      {0,   255, 255}, true}, 
    {7,  "Mud",      {139, 69,  19 }, true}, 
    {8,  "Forest",   {0,   100, 0  }, true}, 
    {9,  "Desert",   {255, 165, 0  }, true}, 
    {10, "Mountain", {105, 105, 105}, false}, 
    {11, "Swamp",    {46,  139, 87 }, true}, 
    {12, "Jungle",   {34,  139, 34 }, true}, // looks good in my option
    {13, "Tundra",   {0,   128, 128}, true}, 
    {14, "Savanna",  {218, 165, 32 }, true}, 
    {15, "Taiga",    {0,   128, 0  }, true}, 
    {16, "Steppe",   {210, 180, 140}, true}, 
    {17, "Prairie",  {255, 228, 196}, true}, 
    {18, "Plateau",  {205, 133, 63 }, true}, 
    {19, "Canyon",   {160, 82,  45 }, false}, 
    {20, "Badlands", {178, 34,  34 }, true}, 
    {21, "Mesa",     {222, 184, 135}, false}, 
    {22, "Oasis",    {0,   255, 127}, true}, 
    {23, "Volcano",  {128, 0,   0  }, false}, 
    {24, "Glacier",  {176, 224, 230}, false}, 
    {25, "Fjord",    {70,  130, 180}, false}, 
    {26, "Bay",      {0,   105, 148}, false}, 
    {27, "Cove",     {100, 149, 237}, false}, 
    {28, "Beach",    {238, 214, 175}, true}, 
    {29, "Cliff",    {112, 128, 144}, false}, 
    {30, "Dune",     {194, 178, 128}, true}  
};

// Function to initialize the constant memory
void TerrainTypes::initializeTerrainTypes() {
    cudaMemcpyToSymbol(TERRAINS, terrainData, sizeof(TerrainType) * 31);
}

// Implementation of getTerrainById
__device__ __host__ const TerrainType* TerrainTypes::getTerrainById(int id) {
    return (id >= 0 && id <= DUNE) ? &TERRAINS[id] : nullptr;
}