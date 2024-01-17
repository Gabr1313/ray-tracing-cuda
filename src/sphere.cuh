#pragma once

#include "algebra.cuh"
#include "ray.cuh"

typedef struct _Sphere {
	Float3 center;
	float radius;
} Sphere;

Sphere sphere_new(const Float3* center, const float radius);

__device__ __host__ float sphere_intersect_distance(const void* sphere, const Ray3* ray);
__device__ __host__ Float3 sphere_normal_normalized(const void* sphere, const Float3* point);
