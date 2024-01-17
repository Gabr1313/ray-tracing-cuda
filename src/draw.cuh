#pragma once

#include "algebra.cuh"
#include "scanner.cuh"

void shoot_and_draw(const InputData* input_data);
__device__ Float3 trace_ray(const Ray3* ray, const Object* object_ptr,
							const int number_of_objects, const int max_bounces,
							const Float3* background, curandState* state);
