@echo off
cd CPU
echo ========== TerrainCraft CPU Minimal Build Script ==========

:: Create bin directory if it doesn't exist
echo [1/5] Creating bin directory if it doesn't exist...
if not exist bin mkdir bin

:: Clean any previous build artifacts
echo [2/5] Cleaning previous build artifacts...
if exist bin\minimal.exe del /f bin\minimal.exe

:: Compile with g++ - minimal build with debug flags
echo [3/5] Compiling minimal CPU implementation...
g++ -std=c++11 -g -O0 ^
src/main.cpp ^
src/terrain/terrain_types.cpp ^
src/terrain/terrain_generator.cpp ^
src/visualization/visualization.cpp ^
src/noise/perlin_noise.cpp ^
src/noise/voronoi_noise.cpp ^
src/noise/noise_utils.cpp ^
-o bin/minimal

:: Check if compilation succeeded
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed with error code %errorlevel%
    exit /b %errorlevel%
)

echo [4/5] Compilation successful!

:: Run the program with a very small test size
echo [5/5] Running minimal program...
cd bin

:: Run with very small size (16x16)
echo Running with minimal size (16x16)...
minimal.exe 16

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Program execution failed with error code %ERRORLEVEL%
    cd ..
    exit /b %ERRORLEVEL%
)

echo.
echo Test run succeeded!

cd ..
echo ========== Minimal Build and Run Complete ==========