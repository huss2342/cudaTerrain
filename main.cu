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

// ======================== Perlin Noise Functions ========================
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

__device__ float noise(float x, float y, float z) {
    // Find unit cube that contains the point
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;
    int Z = (int)floorf(z) & 255;
    
    // Find relative x, y, z of point in cube
    x -= floorf(x);
    y -= floorf(y);
    z -= floorf(z);
    
    // Compute fade curves for each of x, y, z
    float u = fade(x);
    float v = fade(y);
    float w = fade(z);
    
    // Hash coordinates of the 8 cube corners
    int A = permutation[X] + Y;
    int AA = permutation[A] + Z;
    int AB = permutation[A + 1] + Z;
    int B = permutation[X + 1] + Y;
    int BA = permutation[B] + Z;
    int BB = permutation[B + 1] + Z;
    
    // Add blended results from 8 corners of cube
    return lerp(
        lerp(
            lerp(grad(permutation[AA], x, y, z),
                 grad(permutation[BA], x-1, y, z), u),
            lerp(grad(permutation[AB], x, y-1, z),
                 grad(permutation[BB], x-1, y-1, z), u), v),
        lerp(
            lerp(grad(permutation[AA+1], x, y, z-1),
                 grad(permutation[BA+1], x-1, y, z-1), u),
            lerp(grad(permutation[AB+1], x, y-1, z-1),
                 grad(permutation[BB+1], x-1, y-1, z-1), u), v), w);
}

// A kernel function to generate terrain based on Perlin noise
__global__ void generateTerrain(int* terrain, int width, int height, float scale, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Generate noise value
        float nx = (float)x / width * scale + offsetX;
        float ny = (float)y / height * scale + offsetY;
        // Fixed z value for 2D terrain
        float nz = 0.0f;
        
        // Get noise value between -1 and 1
        float value = noise(nx, ny, nz);
        
        // Scale to 0 to 1 range
        value = (value + 1.0f) * 0.5f;
        
        // Map noise to terrain types (simple example)
        int terrainType;
        if (value < 0.2f) {
            terrainType = WATER;
        } else if (value < 0.3f) {
            terrainType = SAND;
        } else if (value < 0.7f) {
            terrainType = GRASS;
        } else if (value < 0.8f) {
            terrainType = ROCK;
        } else {
            terrainType = SNOW;
        }
        
        // Store terrain type in output array
        terrain[y * width + x] = terrainType;
    }
}


// Helper function to launch the kernel
void createPerlinNoiseTerrain(int* d_terrain, int width, int height, float scale = 8.0f, float offsetX = 0.0f, float offsetY = 0.0f) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    generateTerrain<<<gridSize, blockSize>>>(d_terrain, width, height, scale, offsetX, offsetY);
}
// ======================== Visualize ======================

// Add this function to map Perlin noise to RGB colors for visualization
__global__ void visualizeTerrain(int* terrain, unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int terrainType = terrain[idx];
        
        // Get the color from the terrain type
        const TerrainType* terrainInfo = TerrainTypes::getTerrainById(terrainType);
        
        // Set RGB values in image (assuming 3 channels)
        image[idx * 3 + 0] = terrainInfo->color.r;
        image[idx * 3 + 1] = terrainInfo->color.g;
        image[idx * 3 + 2] = terrainInfo->color.b;
    }
}

// Function to save the image to a PPM file (simple format for testing)
void saveToPPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(image, 3, width * height, fp);
    
    fclose(fp);
    printf("Saved terrain image to %s\n", filename);
}

// ======================== Main ======================
int main() {
    // Define terrain size
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(int);
    int imageSize = width * height * 3 * sizeof(unsigned char); // RGB
    
    // Allocate host memory
    int* h_terrain = (int*)malloc(size);
    unsigned char* h_image = (unsigned char*)malloc(imageSize);
    
    // Allocate device memory
    int* d_terrain;
    unsigned char* d_image;
    cudaMalloc(&d_terrain, size);
    cudaMalloc(&d_image, imageSize);
    
    // Generate terrain
    createPerlinNoiseTerrain(d_terrain, width, height);
    
    // Visualize terrain
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    visualizeTerrain<<<gridSize, blockSize>>>(d_terrain, d_image, width, height);
    
    // Copy results back to host
    cudaMemcpy(h_terrain, d_terrain, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    
    // Save image to file
    saveToPPM("terrain.ppm", h_image, width, height);
    
    // Clean up
    free(h_terrain);
    free(h_image);
    cudaFree(d_terrain);
    cudaFree(d_image);
    
    return 0;
}