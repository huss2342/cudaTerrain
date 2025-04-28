#!/bin/bash
echo "========== TerrainCraft CPU Build Script =========="

# Create bin directory if it doesn't exist
echo "[1/5] Creating bin directory if it doesn't exist..."
cd CPU
if [ ! -d "bin" ]; then
    mkdir bin
fi

# Clean any previous build artifacts
echo "[2/5] Cleaning previous build artifacts..."
if [ -f "bin/main" ]; then
    rm -f bin/main
fi

# Compile with optimizations
echo "[3/5] Compiling C++ files with optimizations enabled..."
g++ -o bin/main \
    src/main.cpp \
    src/noise/perlin_noise.cpp \
    src/noise/voronoi_noise.cpp \
    src/noise/noise_utils.cpp \
    src/terrain/terrain_generator.cpp \
    src/terrain/terrain_smoothing.cpp \
    src/terrain/terrain_types.cpp \
    src/terrain/terrain_height.cpp \
    src/visualization/visualization.cpp \
    -I include \
    -O3

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
echo "========== Build and Run Complete =========="