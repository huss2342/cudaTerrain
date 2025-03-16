#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

// ======================== Perlin Noise ========================
/*
https://adrianb.io/2014/08/09/perlinnoise.html
Basic kernel that generates perlin noise
maps the noise values to terrain types
to be visualized later
*/

__device__ __constant__ int permutation[256] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

__device__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float grad(int hash, float x, float y, float z) {
    switch(hash & 0xF)
    {
        case 0x0: return  x + y;
        case 0x1: return -x + y;
        case 0x2: return  x - y;
        case 0x3: return -x - y;
        case 0x4: return  x + z;
        case 0x5: return -x + z;
        case 0x6: return  x - z;
        case 0x7: return -x - z;
        case 0x8: return  y + z;
        case 0x9: return -y + z;
        case 0xA: return  y - z;
        case 0xB: return -y - z;
        case 0xC: return  y + x;
        case 0xD: return -y + z;
        case 0xE: return  y - x;
        case 0xF: return -y - z;
        default: return 0; // never happens
    }
}

// ======================== Main ======================
int main() {
    for (int i = 0; i <= DUNE; i++) {
        const TerrainType* terrain = TerrainTypes::getTerrainById(i);
        printf("terrain: %s (RGB: %d,%d,%d)\n", 
            terrain->name, terrain->color.r, terrain->color.g, terrain->color.b);
    }
    return 0;
}