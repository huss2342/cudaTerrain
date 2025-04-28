#!/bin/bash
cd GPU1
echo "========== TerrainCraft Build Script =========="

# Create bin directory if it doesn't exist
echo "[1/5] Creating bin directory if it doesn't exist..."
if [ ! -d "bin" ]; then
    mkdir bin
fi

# Clean any previous build artifacts
echo "[2/5] Cleaning previous build artifacts..."
if [ -f "bin/main" ]; then
    rm -f bin/main
fi

# Compile with relocatable device code
echo "[3/5] Compiling CUDA files with relocatable device code enabled..."
/usr/local/cuda/bin/nvcc -rdc=true -G \
    src/main.cu \
    src/terrain/terrain_types.cu \
    src/terrain/terrain_generator.cu \
    src/terrain/terrain_smoothing.cu \
    src/terrain/component_analysis.cu \
    src/noise/perlin_noise.cu \
    src/noise/voronoi_noise.cu \
    src/noise/noise_utils.cu \
    src/visualization/visualization.cu \
    src/terrain/terrain_height.cu \
    -o bin/main

# Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Compilation failed with error code $?"
    exit $?
fi

echo "[4/5] Compilation successful!"

# Run the program
echo "[5/5] Running the program..."
cd bin
./main 80 4096
cd ..

cd ..
echo "========== Build and Run Complete =========="s