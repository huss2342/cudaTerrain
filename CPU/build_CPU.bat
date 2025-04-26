@echo off
echo ========== TerrainCraft CPU Build Script ==========

:: Create bin directory if it doesn't exist
echo [1/5] Creating bin directory if it doesn't exist...
cd CPU
if not exist bin mkdir bin

:: Clean any previous build artifacts
echo [2/5] Cleaning previous build artifacts...
if exist bin\main.exe del /f bin\main.exe
if exist bin\main.lib del /f bin\main.lib
if exist bin\main.exp del /f bin\main.exp

:: Compile with optimizations
echo [3/5] Compiling C++ files with optimizations enabled...
g++ -o bin/main.exe ^
    src/main.cpp ^
    src/noise/perlin_noise.cpp ^
    src/noise/voronoi_noise.cpp ^
    src/noise/noise_utils.cpp ^
    src/terrain/terrain_generator.cpp ^
    src/terrain/terrain_smoothing.cpp ^
    src/terrain/terrain_types.cpp ^
    src/terrain/terrain_height.cpp ^
    src/visualization/visualization.cpp ^
    -I include ^
    -O3

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

cd ..
echo ========== Build and Run Complete ==========