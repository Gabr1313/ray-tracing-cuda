#include "draw.cuh"
#include "scanner.cuh"

int main() {
	const InputData input_data = scan_input();
	shoot_and_draw(&input_data);
	return 0;
}
