@echo off
cd CPU
echo ========== TerrainCraft CPU Build Script ==========

:: Create bin directory if it doesn't exist
echo [1/5] Creating bin directory if it doesn't exist...
if not exist bin mkdir bin

:: Clean any previous build artifacts
echo [2/5] Cleaning previous build artifacts...
if exist bin\main.exe del /f bin\main.exe

:: Compile with g++ (MinGW) or MSVC
echo [3/5] Compiling CPU files...
g++ -std=c++11 -O3 -pthread ^
src/main.cpp ^
src/terrain/terrain_types.cpp ^
src/terrain/terrain_generator.cpp ^
src/terrain/terrain_smoothing.cpp ^
src/terrain/component_analysis.cpp ^
src/noise/perlin_noise.cpp ^
src/noise/voronoi_noise.cpp ^
src/noise/noise_utils.cpp ^
src/visualization/visualization.cpp ^
src/terrain/terrain_height.cpp ^
-o bin/main

:: Check if compilation succeeded
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed with error code %errorlevel%
    exit /b %errorlevel%
)

echo [4/5] Compilation successful!

:: Run the program
echo [5/5] Running the program...
cd bin
main.exe 80 4096
cd ..

echo ========== Build and Run Complete ==========