#include "../../include/terrain/terrain_types.h"
#include <iostream>

namespace TerrainTypes {
    TerrainType TERRAINS[31];

    void initializeTerrainTypes() {
        // Initialize all terrain types with their proper colors
        TERRAINS[WATER]    = {WATER,    "Water",    {0,   0,   255}, false};
        TERRAINS[SAND]     = {SAND,     "Sand",     {255, 235, 130}, true};
        TERRAINS[GRASS]    = {GRASS,    "Grass",    {0,   200, 0  }, true};
        TERRAINS[ROCK]     = {ROCK,     "Rock",     {128, 128, 128}, true};
        TERRAINS[SNOW]     = {SNOW,     "Snow",     {255, 255, 255}, true};
        TERRAINS[LAVA]     = {LAVA,     "Lava",     {255, 0,   0  }, false};
        TERRAINS[ICE]      = {ICE,      "Ice",      {0,   255, 255}, true};
        TERRAINS[MUD]      = {MUD,      "Mud",      {139, 69,  19 }, true};
        TERRAINS[FOREST]   = {FOREST,   "Forest",   {0,   100, 0  }, true};
        TERRAINS[DESERT]   = {DESERT,   "Desert",   {255, 165, 0  }, true};
        TERRAINS[MOUNTAIN] = {MOUNTAIN, "Mountain", {105, 105, 105}, false};
        TERRAINS[SWAMP]    = {SWAMP,    "Swamp",    {46,  139, 87 }, true};
        TERRAINS[JUNGLE]   = {JUNGLE,   "Jungle",   {34,  139, 34 }, true};
        TERRAINS[TUNDRA]   = {TUNDRA,   "Tundra",   {0,   128, 128}, true};
        TERRAINS[SAVANNA]  = {SAVANNA,  "Savanna",  {218, 165, 32 }, true};
        TERRAINS[TAIGA]    = {TAIGA,    "Taiga",    {0,   128, 0  }, true};
        TERRAINS[STEPPE]   = {STEPPE,   "Steppe",   {210, 180, 140}, true};
        TERRAINS[PRAIRIE]  = {PRAIRIE,  "Prairie",  {255, 228, 196}, true};
        TERRAINS[PLATEAU]  = {PLATEAU,  "Plateau",  {205, 133, 63 }, true};
        TERRAINS[CANYON]   = {CANYON,   "Canyon",   {160, 82,  45 }, false};
        TERRAINS[BADLANDS] = {BADLANDS, "Badlands", {178, 34,  34 }, true};
        TERRAINS[MESA]     = {MESA,     "Mesa",     {222, 184, 135}, false};
        TERRAINS[OASIS]    = {OASIS,    "Oasis",    {0,   255, 127}, true};
        TERRAINS[VOLCANO]  = {VOLCANO,  "Volcano",  {128, 0,   0  }, false};
        TERRAINS[GLACIER]  = {GLACIER,  "Glacier",  {176, 224, 230}, false};
        TERRAINS[FJORD]    = {FJORD,    "Fjord",    {70,  130, 180}, false};
        TERRAINS[BAY]      = {BAY,      "Bay",      {0,   105, 148}, false};
        TERRAINS[COVE]     = {COVE,     "Cove",     {100, 149, 237}, false};
        TERRAINS[BEACH]    = {BEACH,    "Beach",    {238, 214, 175}, true};
        TERRAINS[CLIFF]    = {CLIFF,    "Cliff",    {112, 128, 144}, false};
        TERRAINS[DUNE]     = {DUNE,     "Dune",     {194, 178, 128}, true};
    }

    const TerrainType* getTerrainById(int id) {
        if (id < 0 || id > DUNE) {
            std::cerr << "Invalid terrain ID: " << id << std::endl;
            return &TERRAINS[GRASS]; // Return grass as default
        }
        return &TERRAINS[id];
    }
}