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
src/main.cu src/terrain_types.cu src/terrain_gen.cu ^
src/perlin_noise.cu src/visualization.cu -o bin/main


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
main.exe 4.0 4096
cd ..

echo ========== Build and Run Complete ==========