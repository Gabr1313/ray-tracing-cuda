#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdio>

#include "algebra.cuh"
#include "object.cuh"

Object object_new(const int shape_type, const Shape* shape, const Float3* color,
				  const float emission_intensity, const float reflection) {
	static int id = 0;
	Object object;
	object.id = id++;
	object.shape = *shape;
	object.color = *color;
	object.light_emitted = float3_mul(color, emission_intensity);
	object.reflection = reflection;
	object.shape_type = shape_type;
	return object;
}

__device__ __host__ float object_intersect_distance(const Object* object,
													const Ray3* ray) {
	// switch (object->shape_type) {
	// 	case TYPE_SPHERE:
	// 		return sphere_intersect_distance(&object->shape, ray);
	// 	case TYPE_PLANE:
	// 		return plane_intersect_distance(&object->shape, ray);
	// 	case TYPE_TRIANGLE:
	// 		return triangle_intersect_distance(&object->shape, ray);
	// }
	// return -1;

	if (object->shape_type == TYPE_SPHERE) {
		return sphere_intersect_distance(&object->shape, ray);
	} else if (object->shape_type == TYPE_PLANE) {
		return plane_intersect_distance(&object->shape, ray);
	} else /* if (object->shape_type == TYPE_TRIANGLE)  */ {
		return triangle_intersect_distance(&object->shape, ray);
	}
}

__device__ __host__ Float3 object_normal_normalized(const Object* object,
													const Ray3* ray) {
	Float3 direction;

	// switch (object->shape_type) {
	// 	case TYPE_SPHERE:
	// 		direction = sphere_normal_normalized(&object->shape, &ray->origin);
	// 	case TYPE_PLANE:
	// 		direction = plane_normal_normalized(&object->shape, &ray->origin);
	// 	case TYPE_TRIANGLE:
	// 		direction =
	// 			triangle_normal_normalized(&object->shape, &ray->origin);
	// }

	if (object->shape_type == TYPE_SPHERE)
		direction = sphere_normal_normalized(&object->shape, &ray->origin);
	else if (object->shape_type == TYPE_PLANE)
		direction = plane_normal_normalized(&object->shape, &ray->origin);
	else /* if (object->shape_type == TYPE_TRIANGLE) */
		direction = triangle_normal_normalized(&object->shape, &ray->origin);

	if (float3_dot(&ray->direction, &direction) > 0.0)
		float3_invert_eq(&direction);
	return direction;
}

ObjectVec objectvec_new(const int n) {
	ObjectVec obj_container;
	obj_container.size = 0;
	obj_container.capacity = n;
	obj_container.ptr = (Object*)malloc(sizeof(Object) * n);
	if (obj_container.ptr == NULL) {
		fprintf(stderr, "Error: malloc failed in new_object_container()\n");
		exit(-1);
	}
	return obj_container;
}

void object_vec_push(ObjectVec* object_v, const Object* object) {
	if (object_v->size >= object_v->capacity) {
		fprintf(stderr, "Error: object_v->size >= object_v->capacity\n");
		exit(-1);
	}
	memcpy(object_v->ptr + object_v->size, object, sizeof(Object));
	object_v->size++;
}

__device__ void object_reflect_ray(const Object* object, Ray3* ray,
								   const float distance, curandState* state) {
	ray3_move_along(ray, distance);
	const Float3 normal = object_normal_normalized(object, ray);
	const float flip = (float)curand(state) / 4294967295.0f;
	if (flip < object->reflection) {
		ray->direction = float3_mirror(&ray->direction, &normal);
	} else {
		ray->direction = half_sphere_random(&normal, state);
	}
}

#define PI 3.14159265358979323846

__device__ Float3 half_sphere_random(const Float3* normal, curandState* state) {
	const float phi = 2 * PI * (float)curand(state) / 4294967295.0f;
	const float theta = PI * (float)curand(state) / 4294967295.0f;
	const float sin_theta = sinf(theta);
	Float3 retval =
		float3_new(sin_theta * cosf(phi), sin_theta * sinf(phi), cosf(theta));
	if (float3_dot(&retval, normal) < 0) {
		float3_invert_eq(&retval);
	}
	return retval;
}
