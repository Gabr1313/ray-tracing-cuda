#pragma once

#include "algebra.cuh"
#include "ray.cuh"

typedef struct _Plane {
	Float3 normal;
	float d;
} Plane;

Plane plane_new(const float a, const float b, const float c, const float d);
Plane plane_from_points(const Float3* p1, const Float3* p2, const Float3* p3);
__device__ __host__ float plane_intersect_distance(const void* plane, const Ray3* ray);
__device__ __host__ Float3 plane_normal_normalized(const void* plane, const Float3* point);
