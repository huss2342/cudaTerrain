@echo off
setlocal enabledelayedexpansion

cd CPU

:: Create bin directory if it doesn't exist
if not exist bin mkdir bin

:: Clean previous build
del /Q bin\*

:: Compile the CPU implementation
g++ -o bin/minimal.exe ^
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

:: Check if compilation was successful
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

:: Run the program with scale=80 and size=4096
bin\minimal.exe 80 4096

:: Check if program execution was successful
if %ERRORLEVEL% NEQ 0 (
    echo Program execution failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

cd ..
endlocal