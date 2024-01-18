#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cstdio>

#include "algebra.cuh"
#include "draw.cuh"
#include "object.cuh"
#include "ray.cuh"
#include "scanner.cuh"

__global__ void shoot(const InputData* input_data, unsigned char* buffer,
					  Object* object_ptr);

__device__ __host__ int local_min(float a, float b) {
	return (int)(a < b ? a : b);
}

void shoot_and_draw(const InputData* input_data) {
	const int width = input_data->camera.width;
	const int height = input_data->camera.height;
	const int total_pixels = width * height;
	const int content_len = total_pixels * 3;
	const Object* object_ptr = input_data->objects.ptr;

	InputData* input_data_cuda;

	unsigned char* buffer =
		(unsigned char*)malloc(content_len * sizeof(unsigned char));
	unsigned char* buffer_cuda;
	Object* object_ptr_cuda;
	cudaMalloc((void**)&input_data_cuda, sizeof(InputData));
	cudaMalloc((void**)&buffer_cuda, content_len * sizeof(unsigned char));
	cudaMalloc((void**)&object_ptr_cuda,
			   sizeof(Object) * input_data->objects.size);
	if (buffer_cuda == NULL || input_data_cuda == NULL ||
		object_ptr_cuda == NULL) {
		fprintf(stderr, "Error: can't allocate GPU memory.\n");
		exit(-1);
	}
	cudaMemcpy(input_data_cuda, input_data, sizeof(InputData),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(object_ptr_cuda, object_ptr,
			   sizeof(Object) * input_data->objects.size,
			   cudaMemcpyHostToDevice);

	int threads = 256;
	int blocks = (total_pixels + threads - 1) / threads;
	shoot<<<blocks, threads>>>(input_data_cuda, buffer_cuda, object_ptr_cuda);
	// cudaDeviceSynchronize();
	cudaFree(input_data_cuda);
	cudaFree(object_ptr_cuda);
	cudaMemcpy(buffer, buffer_cuda, content_len * sizeof(unsigned char),
			   cudaMemcpyDeviceToHost);
	cudaFree(buffer_cuda);

	char header[64];
	sprintf(header, "P6\n%d %d\n255\n", width, height);
	const int heder_len = strlen(header);
	fwrite(header, sizeof(char), heder_len, stdout);
	fwrite(buffer, sizeof(unsigned char), content_len, stdout);

	free(buffer);
	free(input_data->objects.ptr);
}

__global__ void shoot(const InputData* input_data, unsigned char* buffer,
					  Object* object_ptr) {
	const int width = input_data->camera.width;
	const int height = input_data->camera.height;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int number_of_updates = input_data->number_of_updates;
	const int number_of_objects = input_data->objects.size;
	if (idx >= width * height) return;
	const int i = idx / width;
	const int j = idx % width;
	const int sqrt_ray_per_pixel = input_data->camera.sqrt_ray_per_pixel;
	const int max_bounces = input_data->max_bounces;
	const Ray3* upper_left = &input_data->camera.upper_left;
	const Float3* delta_x = &input_data->camera.delta_x;
	const Float3* delta_y = &input_data->camera.delta_y;
	const Float3* d_x = &input_data->camera.d_x;
	const Float3* d_y = &input_data->camera.d_y;
	const Float3* background = &input_data->background_color;
	curandState state;
	curand_init(idx, 0, 0, &state);

	Ray3 ray = *upper_left;
	const Float3 diff_y = float3_mul(delta_y, i);
	const Float3 diff_x = float3_mul(delta_x, j);
	float3_add_eq(&ray.direction, &diff_y);
	float3_add_eq(&ray.direction, &diff_x);

	Float3 color_sum = float3_new(0, 0, 0);
	Float3 starting_direction_y = ray.direction;
	for (int ii = 0; ii < sqrt_ray_per_pixel; ii++) {
		ray.direction = starting_direction_y;
		for (int jj = 0; jj < sqrt_ray_per_pixel; jj++) {
			for (int nou = 0; nou < number_of_updates; nou++) {
				Float3 light = trace_ray(&ray, object_ptr, number_of_objects,
										 max_bounces, background, &state);
				float3_add_eq(&color_sum, &light);
			}
			float3_add_eq(&ray.direction, d_x);
		}
		float3_add_eq(&starting_direction_y, d_y);
	}

	idx *= 3;
	const float to_multiply =
		255.0f / (number_of_updates * sqrt_ray_per_pixel * sqrt_ray_per_pixel);
	buffer[idx] = local_min(color_sum.x * to_multiply, 255);
	buffer[++idx] = local_min(color_sum.y * to_multiply, 255);
	buffer[++idx] = local_min(color_sum.z * to_multiply, 255);
}

__device__ Float3 trace_ray(const Ray3* ray, const Object* object_ptr,
							const int number_of_objects, const int max_bounces,
							const Float3* background, curandState* state) {
	Float3 color = float3_new(1, 1, 1);
	Float3 light = float3_new(0, 0, 0);
	Ray3 local_ray = *ray;
	const Object* prev = NULL;
	for (int i = 0; i < max_bounces; i++) {
		const Object* nearest_object = NULL;
		float nearest_distance = INFINITY;
		for (int j = 0; j < number_of_objects; j++) {
			const Object* object_found = &object_ptr[j];
			float distance_found =
				object_intersect_distance(object_found, &local_ray);
			if (distance_found > 0 && prev != object_found &&
				distance_found < nearest_distance) {
				nearest_object = object_found;
				nearest_distance = distance_found;
			}
		}
		if (nearest_object == NULL) {
			const Float3 added_light = float3_mul_float3(background, &color);
			float3_add_eq(&light, &added_light);
			break;
		} else {
			prev = nearest_object;
			object_reflect_ray(nearest_object, &local_ray, nearest_distance,
							   state);
			const Float3 added_light =
				float3_mul_float3(&nearest_object->light_emitted, &color);
			float3_add_eq(&light, &added_light);
			float3_mul_eq_float3(&color, &nearest_object->color);
		}
	}
	return light;
}
