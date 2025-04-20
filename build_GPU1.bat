@echo off

echo ========== TerrainCraft Build Script ==========

:: Create bin directory if it doesn't exist
echo [1/5] Creating bin directory if it doesn't exist...
if not exist GPU1\bin mkdir GPU1\bin

:: Clean any previous build artifacts
echo [2/5] Cleaning previous build artifacts...
if exist GPU1\bin\main.exe del /f GPU1\bin\main.exe
if exist GPU1\bin\main.lib del /f GPU1\bin\main.lib
if exist GPU1\bin\main.exp del /f GPU1\bin\main.exp

:: Compile with relocatable device code
echo [3/5] Compiling CUDA files with relocatable device code enabled...
nvcc -rdc=true -G ^
GPU1/src/main.cu ^
GPU1/src/terrain/terrain_types.cu ^
GPU1/src/terrain/terrain_generator.cu ^
GPU1/src/terrain/terrain_smoothing.cu ^
GPU1/src/terrain/component_analysis.cu ^
GPU1/src/noise/perlin_noise.cu ^
GPU1/src/noise/voronoi_noise.cu ^
GPU1/src/noise/noise_utils.cu ^
GPU1/src/visualization/visualization.cu ^
GPU1/src/terrain/terrain_height.cu ^
-o GPU1/bin/main


:: Check if compilation succeeded
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed with error code %errorlevel%
    exit /b %errorlevel%
)

echo [4/5] Compilation successful!

:: Run the program
echo [5/5] Running the program...
cd GPU1\bin
main.exe 80 4096
cd ..

echo ========== Build and Run Complete ==========