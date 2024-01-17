#pragma once

#include "algebra.cuh"

typedef struct _Ray3 {
	Float3 origin, direction;
} Ray3;

__host__ __device__ Ray3 ray3_new(const Float3* position, const Float3* direction);
__host__ __device__ void ray3_move_along(Ray3* ray, const float distance);

typedef struct _Ray2 {
	Float2 origin, direction;
} Ray2;

__host__ __device__ Ray2 ray2_new(const Float2* origin, const Float2* direction);
