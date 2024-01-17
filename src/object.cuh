#pragma once

#include <curand_kernel.h>

#include "algebra.cuh"
#include "plane.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "triangle.cuh"

#define TYPE_SPHERE 1
#define TYPE_PLANE 2
#define TYPE_TRIANGLE 3

typedef union _Shape {
	Sphere sphere;
	Plane plane;
	Triangle triangle;
} Shape;

typedef struct _Object {
	int shape_type, id;
	Shape shape;
	Float3 color, light_emitted;
	float reflection;
} Object;

Object object_new(const int shape_id, const Shape* shape, const Float3* color,
				  const float emission_intensity, const float reflection);
__device__ __host__ float object_intersect_distance(const Object* object,
													const Ray3* ray);
__device__ __host__ Float3 object_normal_normalized(const Object* object,
													const Ray3* ray);
__device__ void object_reflect_ray(const Object* object, Ray3* ray,
								   const float distance, curandState* states);
__device__ Float3 half_sphere_random(const Float3*, curandState* states);

typedef struct _ObjectContainer {
	Object* ptr;
	int size, capacity;
} ObjectVec;

ObjectVec objectvec_new(const int n);
void object_vec_push(ObjectVec* object_v, const Object* object);
