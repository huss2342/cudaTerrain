# TerrainCraft: GPU-Accelerated Procedural Terrain Generation

A CUDA C++ implementation of procedural terrain generation using parallel computing.

## Features

Current:
- Perlin noise terrain generation
- Voronoi biome distribution
- 31 different terrain types with distinct colors
- PPM image output for visualization

Next steps:
- Height dimension (2.5D)
- Landmarks (flags, ruins, etc.)
- Water features (rivers, lakes)
- Chunked map generation

## Requirements

- CUDA Toolkit
- Visual Studio with C++ compiler (Windows)

## Building & Running (Windows)

1. Setup compiler (one-time):
   - Run PowerShell as Administrator
   - Run: `.\add_cl_to_path.ps1`

2. Build and run:
```
.\build.bat
```

The program generates a terrain map and saves it as a .ppm file. Default output is `terrain_scale1.0_size4096.ppm`.

## Command-line Options
```
main.exe <scale> <map_size>
```
- `scale`: Controls terrain feature size (default: 1.0)
  - Lower values = larger features
  - Higher values = smaller features
- `size`: Map dimensions in pixels (default: 4096)

Example: `main.exe 2.5 1024` creates a 1024×1024 map with smaller features.

## Algorithms & CUDA Kernels

### Noise Generation
- **Perlin Noise**: Multi-octave gradient noise for smooth terrain elevation
  - `__device__ float noise(float x, float y, float z)`: GPU device function for basic Perlin noise
  - `__device__ float enhancedNoise(float x, float y, float z)`: GPU device function for improved distribution
  - `__device__ float distributedNoise(float x, float y, float z, int octaves)`: GPU device function for multi-octave fractal noise

- **Voronoi Noise**: Cell-based noise for biome separation
  - `__device__ float voronoiNoise(float x, float y, int seed)`: GPU device function for distance-based cell noise

### Terrain Generation
- **Primary Generation**:
  - `generateTerrain<<<gridSize, blockSize>>>(terrain, width, height, scale, offsetX, offsetY)`: CUDA kernel for terrain type generation using combined noise algorithms
  - `void createPerlinNoiseTerrain(int* d_terrain, int width, height, scale, offsetX, offsetY)`: Host function orchestrating the terrain creation pipeline

### Terrain Processing & Refinement
- **Smoothing Operations**:
  - `smoothTerrain<<<gridSize, blockSize>>>(terrain, output, width, height)`: CUDA kernel for basic terrain smoothing
  - `improvedSmoothTerrain<<<gridSize, blockSize>>>(terrain, output, width, height)`: CUDA kernel for enhanced smoothing with wider radius
  - `removeVerticalStripes<<<gridSize, blockSize>>>(terrain, output, width, height)`: CUDA kernel that eliminates artifacts from noise patterns
  - `removeIsolatedNoise<<<gridSize, blockSize>>>(terrain, output, width, height)`: CUDA kernel that removes single-pixel noise
  - `cleanupSmallPatches<<<gridSize, blockSize>>>(terrain, output, width, height, minRegionSize)`: CUDA kernel that merges small regions

- **Component Analysis**:
  - `identifyConnectedComponents<<<gridSize, blockSize>>>(terrain, labels, width, height)`: CUDA kernel that labels connected regions
  - `propagateLabels<<<gridSize, blockSize>>>(terrain, labels, width, height, changed)`: CUDA kernel that merges component labels
  - `removeSmallComponents<<<gridSize, blockSize>>>(terrain, labels, output, componentSizes, minSize, width, height)`: CUDA kernel that eliminates small components

### Visualization
- `visualizeTerrain<<<gridSize, blockSize>>>(terrain, image, width, height)`: CUDA kernel that converts terrain data to RGB image

## Project Structure

- `include/`: Header files organized by feature
- `src/`: Implementation files for all features
- `bin/`: Build output directory

## Development Phases

🟩 Phase 1: Basic 2D terrain with Perlin noise<br>
🟩 Phase 2: Multiple terrain types and coloring<br>
🟩 Phase 4: Add complex algorithms (Voronoi for biomes)<br>
🟩 Phase 5: Add height dimension (2.5D)<br>


🟥 Phase 6: Implement landmarks<br>
🟥 Phase 7: Add water features (rivers, lakes)<br>
🟥 Phase 8: Implement chunked generation<br>
🟥 Phase 9: Add player and pathfinding<br>
🟥 Phase 10: Advanced features and polish


## Contributors

- Hussein Aljorani
- Ran Duan
