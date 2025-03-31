# TerrainCraft: GPU-Accelerated Procedural Terrain Generation

A CUDA C++ implementation of procedural 2.5D terrain generation leveraging parallel computing for efficient and diverse terrain creation.

## Project Overview

TerrainCraft is a GPU-accelerated terrain generation system designed for:
- Video games
- Simulation environments
- Visualization tools

This project aims to demonstrate efficient procedural terrain generation using CUDA C++ with a focus on performance optimization and scalability for interactive applications.

## Features

Current features:
- Basic 2D terrain generation using Perlin noise
- Multiple terrain types with distinct coloring
- PPM image output for visualization

Planned features:
- Cellular automata for cave generation and natural features
- Voronoi diagrams for biome distribution
- Walkability layer
- Z-dimension implementation (2.5D)
- Landmarks and advanced features (rivers, roads)
- Optional pathfinding implementation (A*)

## Requirements

- CUDA Toolkit
- C++ compiler compatible with CUDA (Visual Studio recommended for Windows)
- CMake (optional, build system not yet implemented)

## Building the Project

### Windows Setup

1. Install CUDA Toolkit and Visual Studio with C++ development tools

2. Add the Visual Studio compiler to your PATH (required for NVCC):
   - Right-click on PowerShell and select "Run as Administrator"
   - Navigate to the project directory
   - Run the setup script:
     ```
     .\add_cl_to_path.ps1
     ```
   - This script will find the Visual Studio cl.exe compiler and add it to your PATH

3. Build the project using the provided build script:
   ```
   .\build.bat
   ```
   
   This script will:
   - Create a bin directory if it doesn't exist
   - Clean previous build artifacts
   - Compile with relocatable device code enabled
   - Run the program automatically after successful compilation

### Linux/macOS Setup

1. Install CUDA Toolkit and required dependencies
   ```
   # Ubuntu
   sudo apt-get install build-essential
   
   # macOS (requires CUDA-compatible hardware/drivers)
   brew install cmake
   ```

2. Build the project (NOT TESTED, but something like this):
   ```
   nvcc src/*.cu -o main -I include/
   ```

## Running the Project

### Windows
The build.bat script will automatically run the program after compilation. If you want to run it separately:

```
cd bin
.\main.exe
```

### Linux/macOS
Execute the compiled binary:
```
./main
```

The program will generate a terrain map and save it as `terrain.ppm` in the bin directory (or current directory for Linux/macOS). You can open PPM files with image viewers like GIMP, Photoshop, or online converters.

## Project Structure

- `include/`: Header files
  - `perlin_noise.h`: Perlin noise functions
  - `terrain_gen.h`: Terrain generation
  - `terrain_types.h`: Terrain type definitions
  - `visualization.h`: Visualization functions
- `src/`: Source files
  - `main.cu`: Entry point
  - `perlin_noise.cu`: Perlin noise implementation
  - `terrain_gen.cu`: Terrain generation algorithms
  - `terrain_types.cu`: Terrain type definitions
  - `visualization.cu`: PPM output functions
- `bin/`: Build output directory (created by build script)
- `build.bat`: Windows build script
- `add_cl_to_path.ps1`: PowerShell script for setting up Visual Studio compiler

## Development Phases

1. ✅ Basic 2D terrain with Perlin noise
2. ⬜ Multiple terrain types and coloring
3. ⬜ Walkability layer
4. ⬜ Complex algorithms (cellular automata, Voronoi)
5. ⬜ Z-dimension implementation
6. ⬜ Landmarks and advanced features

## License



## Contributors

- Hussein Aljorani
- Ran Duan