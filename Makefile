NVCC = nvcc
NVCCFLAGS_COMMON = -dc -std=c++11
NVCCFLAGS_DEBUG = $(NVCCFLAGS_COMMON) -g -G -O0
NVCCFLAGS_RELEASE = $(NVCCFLAGS_COMMON) -O3

MODE ?= debug
ifeq ($(MODE), debug)
$(warning MODE is debug)
	NVCCFLAGS = $(NVCCFLAGS_DEBUG)
else
$(warning MODE is release)
	NVCCFLAGS = $(NVCCFLAGS_RELEASE)
endif

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(filter-out $(SRC_DIR)/*.cuh, $(wildcard $(SRC_DIR)/*.cu))
HEADERS = $(wildcard $(SRC_DIR)/*.cuh)
OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SOURCES))
EXECUTABLE = $(BIN_DIR)/ray-tracer

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@

.PHONY: dir
dir:
	dir -p $(OBJ_DIR) $(BIN_DIR)

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/*

run: $(EXECUTABLE)
	./$(EXECUTABLE) <input.txt >draw.ppm
