@echo off

echo ========== TerrainCraft Build Script ==========

:: Create bin directory if it doesn't exist
echo [1/5] Creating bin directory if it doesn't exist...
if not exist bin mkdir bin

:: Clean any previous build artifacts
echo [2/5] Cleaning previous build artifacts...
if exist bin\main.exe del /f bin\main.exe
if exist bin\main.lib del /f bin\main.lib
if exist bin\main.exp del /f bin\main.exp

:: Compile with relocatable device code
echo [3/5] Compiling CUDA files with relocatable device code enabled...
nvcc -rdc=true -G ^
src/main.cu ^
src/terrain/terrain_types.cu ^
src/terrain/terrain_generator.cu ^
src/terrain/terrain_smoothing.cu ^
src/terrain/component_analysis.cu ^
src/noise/perlin_noise.cu ^
src/noise/voronoi_noise.cu ^
src/noise/noise_utils.cu ^
src/visualization/visualization.cu ^
src/terrain/terrain_height.cu ^
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