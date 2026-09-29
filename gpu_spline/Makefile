# Makefile for Portable C++/HIP GPU Cross-Section Spline Engine

CUDA_PATH ?= /usr/local/cuda-12.9
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

ROOT_CFLAGS := $(shell root-config --cflags)
ROOT_LIBS   := $(shell root-config --libs)
XML2_CFLAGS := $(shell xml2-config --cflags)
XML2_LIBS   := $(shell xml2-config --libs)

INCLUDES    := -Iinclude -I$(CUDA_PATH)/include $(XML2_CFLAGS)
CXXFLAGS    := -O3 -std=c++17 -fPIC -fopenmp -Wall $(ROOT_CFLAGS) $(INCLUDES)
NVCCFLAGS   := -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_70 $(INCLUDES)

LDFLAGS     := -Llib -L$(CUDA_PATH)/lib64 $(ROOT_LIBS) $(XML2_LIBS) -lz -lcudart -fopenmp \
               -Wl,-rpath,$(CUDA_PATH)/lib64 -Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN'

OBJ_DIR     := build
BIN_DIR     := bin
LIB_DIR     := lib

OBJS        := $(OBJ_DIR)/gpu_spline_kernel.o \
               $(OBJ_DIR)/gpu_flux_kernel.o \
               $(OBJ_DIR)/gpu_hadron_tensor_kernel.o \
               $(OBJ_DIR)/gpu_qel_integrator.o \
               $(OBJ_DIR)/spline_loader.o \
               $(OBJ_DIR)/gpu_spline_engine.o \
               $(OBJ_DIR)/gpu_spline_c_api.o \
               $(OBJ_DIR)/gpu_flux_preselector.o \
               $(OBJ_DIR)/hadron_tensor_loader.o

TARGET_LIB  := $(LIB_DIR)/libgpuspline.so
TARGET_BIN  := $(BIN_DIR)/benchmark_splines
TARGET_BIN_FLUX := $(BIN_DIR)/benchmark_flux_preselector
TARGET_BIN_HT   := $(BIN_DIR)/benchmark_hadron_tensor

.PHONY: all clean dirs

all: dirs $(TARGET_LIB) $(TARGET_BIN) $(TARGET_BIN_FLUX) $(TARGET_BIN_HT)

# The portability and physics helpers are inline; rebuild their consumers.
$(OBJS): $(wildcard include/gpu_spline/*.h)
$(OBJS): | dirs

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

# Device kernel compilation
$(OBJ_DIR)/gpu_spline_kernel.o: src/gpu_spline_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_flux_kernel.o: src/gpu_flux_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_hadron_tensor_kernel.o: src/gpu_hadron_tensor_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_qel_integrator.o: src/gpu_qel_integrator.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Host C++ compilation
$(OBJ_DIR)/spline_loader.o: src/spline_loader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_spline_engine.o: src/gpu_spline_engine.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_spline_c_api.o: src/gpu_spline_c_api.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_flux_preselector.o: src/gpu_flux_preselector.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/hadron_tensor_loader.o: src/hadron_tensor_loader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Shared Library
$(TARGET_LIB): $(OBJS)
	$(CXX) -shared -o $@ $^ -L$(CUDA_PATH)/lib64 -lcudart $(ROOT_LIBS) $(XML2_LIBS) -lz -fopenmp

# Benchmark binary (splines)
$(TARGET_BIN): benchmark/benchmark_splines.cpp $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lgpuspline $(LDFLAGS)

# Benchmark binary (flux pre-selection)
$(TARGET_BIN_FLUX): benchmark/benchmark_flux_preselector.cpp $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lgpuspline $(LDFLAGS)

# Benchmark binary (hadron tensor)
$(TARGET_BIN_HT): benchmark/benchmark_hadron_tensor.cpp $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lgpuspline $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

# Numerical regressions can run without ROOT/CUDA; the CUDA variant also tests
# nonfatal initialization and, when a device is available, kernel parity.
.PHONY: test test-cpu

test-cpu:
	@mkdir -p $(BIN_DIR)
	$(CXX) -std=c++17 -O2 -DCPU_ONLY -Iinclude -I../src tests/regressions.cpp src/hadron_tensor_loader.cpp -o $(BIN_DIR)/regressions_cpu
	./$(BIN_DIR)/regressions_cpu

test: dirs $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) -I../src tests/regressions.cpp -o $(BIN_DIR)/regressions -L$(LIB_DIR) -lgpuspline $(LDFLAGS)
	./$(BIN_DIR)/regressions

.PHONY: test-python
test-python: dirs $(TARGET_LIB)
	python3 tests/python_regressions.py

# Run after rebuilding the affected GENIE libraries.
.PHONY: test-genie
test-genie: dirs $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) -I../src tests/spline_parity.cpp -o $(BIN_DIR)/spline_parity -Wl,--no-as-needed $(shell ../bin/genie-config --libs) -L$(LIB_DIR) -lgpuspline $(LDFLAGS) $(shell lhapdf-config --ldflags) -lEG -lEGPythia6 -lGeom -lMinuit -lMathMore
	./$(BIN_DIR)/spline_parity
	$(CXX) $(CXXFLAGS) -I../src tests/sampling_bounds.cpp -o $(BIN_DIR)/sampling_bounds -Wl,--no-as-needed $(shell ../bin/genie-config --libs) -L$(LIB_DIR) -lgpuspline $(LDFLAGS) $(shell lhapdf-config --ldflags) -lEG -lEGPythia6 -lGeom -lMinuit -lMathMore
	./$(BIN_DIR)/sampling_bounds

.PHONY: test-qel
test-qel: dirs $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) tests/qel_regressions.cpp -o $(BIN_DIR)/qel_regressions -L$(LIB_DIR) -lgpuspline $(LDFLAGS)
	./$(BIN_DIR)/qel_regressions
