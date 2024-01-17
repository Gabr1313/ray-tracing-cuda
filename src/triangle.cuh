#pragma once

#include "algebra.cuh"
#include "plane.cuh"
#include "ray.cuh"

typedef struct _Triangle {
	Plane plane;
	Ray2 r1, r2, r3;
} Triangle;

Triangle triangle_new(const Float3* p1, const Float3* p2, const Float3* p3);
__device__ __host__ float triangle_intersect_distance(const void* triangle, const Ray3* ray);
__device__ __host__ Float3 triangle_normal_normalized(const void* triangle, const Float3* point);
