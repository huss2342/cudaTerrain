// CPU/src/noise/perlin_noise.cpp - minimal stub implementation
#include "../../include/noise/perlin_noise.h"

// Minimal implementations that just return constant values
float fade(float t) {
    return t;
}

float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float grad(int hash, float x, float y, float z) {
    return 0.0f;
}

float noise(float x, float y, float z) {
    return 0.5f;
}