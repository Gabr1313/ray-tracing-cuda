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

__global__ void shoot(const InputData* input_data, Float3* pixel_sum,
					  Object* object_ptr);

__device__ __host__ int local_min(float a, float b) {
	return (int)(a < b ? a : b);
}

void shoot_and_draw(const InputData* input_data) {
	const int width = input_data->camera.width;
	const int height = input_data->camera.height;
	const int number_of_updates = input_data->number_of_updates;
	const int total_pixels = width * height;
	const int ray_per_pixel = input_data->camera.sqrt_ray_per_pixel *
							  input_data->camera.sqrt_ray_per_pixel;
	const int content_len = total_pixels * 3;
	const Object* object_ptr = input_data->objects.ptr;

	InputData* input_data_cuda;
	Float3* pixel_sum = (Float3*)calloc(width * height, sizeof(Float3));
	Float3* pixel_sum_cuda;
	Object* object_ptr_cuda;
	cudaMalloc((void**)&input_data_cuda, sizeof(InputData));
	cudaMalloc((void**)&pixel_sum_cuda, sizeof(Float3) * total_pixels);
	cudaMalloc((void**)&object_ptr_cuda,
			   sizeof(Object) * input_data->objects.size);
	if (pixel_sum_cuda == NULL) {
		fprintf(stderr, "Error: can't allocate GPU memory for input_data\n");
		exit(-1);
	}
	if (pixel_sum_cuda == NULL) {
		fprintf(stderr, "Error: can't allocate GPU memory for %d pixel\n",
				total_pixels);
		exit(-1);
	}
	cudaMemcpy(input_data_cuda, input_data, sizeof(InputData),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(pixel_sum_cuda, pixel_sum, sizeof(Float3) * total_pixels,
			   cudaMemcpyHostToDevice);
	cudaMemcpy(object_ptr_cuda, object_ptr,
			   sizeof(Object) * input_data->objects.size,
			   cudaMemcpyHostToDevice);

	int threads = 256;
	int blocks = (total_pixels + threads - 1) / threads;
	shoot<<<blocks, threads>>>(input_data_cuda, pixel_sum_cuda,
							   object_ptr_cuda);
	cudaDeviceSynchronize();

	cudaMemcpy(pixel_sum, pixel_sum_cuda, sizeof(Float3) * total_pixels,
			   cudaMemcpyDeviceToHost);

	printf("%d - %f %f %f\n", 0, pixel_sum[0].x, pixel_sum[0].y,
		   pixel_sum[0].z);

	const float to_multiply = 255.0f / (number_of_updates * ray_per_pixel);
	unsigned char* buffer =
		(unsigned char*)malloc(content_len * sizeof(unsigned char));
	if (pixel_sum == NULL) {
		fprintf(stderr, "Error: can't allocate memory for %d pixel\n",
				width * height);
		exit(-1);
	}
	for (int i = 0, j = 0; i < total_pixels; i++) {
		buffer[j++] = local_min(pixel_sum[i].x * to_multiply, 255);
		buffer[j++] = local_min(pixel_sum[i].y * to_multiply, 255);
		buffer[j++] = local_min(pixel_sum[i].z * to_multiply, 255);
	}
	printf("%d - %d %d %d\n", 0, buffer[0], buffer[1], buffer[2]);

	char header[64];
	sprintf(header, "P6\n%d %d\n255\n", width, height);
	const int heder_len = strlen(header);
	fwrite(header, sizeof(char), heder_len, stdout);
	fwrite(buffer, sizeof(unsigned char), content_len, stdout);

	free(buffer);
	free(pixel_sum);
	free(input_data->objects.ptr);
}

__global__ void shoot(const InputData* input_data, Float3* pixel_sum,
					  Object* object_ptr) {
	const int width = input_data->camera.width;
	const int height = input_data->camera.height;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

	Float3 starting_direction = ray.direction;
	for (int nou = 0; nou < number_of_updates; nou++) {
		for (int ii = 0; ii < sqrt_ray_per_pixel; ii++) {
			ray.direction = starting_direction;
			for (int jj = 0; jj < sqrt_ray_per_pixel; jj++) {
				Float3 light = trace_ray(&ray, object_ptr, number_of_objects,
										 max_bounces, background, &state);
				float3_add_eq(&pixel_sum[idx], &light);
				float3_add_eq(&ray.direction, d_x);
			}
			float3_add_eq(&starting_direction, d_y);
		}
	}
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
