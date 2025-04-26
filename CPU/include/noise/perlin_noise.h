#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

float fade(float t);
float lerp(float a, float b, float t);
float grad(int hash, float x, float y, float z);
float noise(float x, float y, float z);

#endif // PERLIN_NOISE_H