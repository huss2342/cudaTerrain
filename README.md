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

## Project Structure

- `include/`: Header files organized by feature
- `src/`: Implementation files for all features
- `bin/`: Build output directory

## Development Phases

[X] Phase 1: Basic 2D terrain with Perlin noise
[X] Phase 2: Multiple terrain types and coloring
[X] Phase 4: Add complex algorithms (Voronoi for biomes)

[ ] Phase 5: Add height dimension (2.5D)
[ ] Phase 6: Implement landmarks
[ ] Phase 7: Add water features (rivers, lakes)
[ ] Phase 8: Implement chunked generation
[ ] Phase 9: Add player and pathfinding
[ ] Phase 10: Advanced features and polish

## Contributors

- Hussein Aljorani
- Ran Duan