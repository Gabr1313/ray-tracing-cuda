#pragma once
#include "algebra.cuh"
#include "camera.cuh"
#include "object.cuh"

typedef struct _InputData {
	int number_of_updates, max_bounces;
	Float3 background_color;
	Camera camera;
	ObjectVec objects;
} InputData;

InputData scan_input();
