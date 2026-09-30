# Makefile for the portable CUDA/HIP GPU library used by GENIE (gpu_spline)
#
#   make                      CUDA build for every architecture in CUDA_ARCHS
#   make CUDA_ARCHS=70        quick build for one architecture
#   make CUDA_ARCHS=native    only the GPU(s) in this machine (HIP_ARCHS=native for AMD)
#   make GPU_BACKEND=hip      AMD ROCm/HIP build for HIP_ARCHS (experimental)
#   make lib                  library only (what GENIE's own build runs)

GPU_BACKEND ?= cuda
CUDA_PATH   ?= /usr/local/cuda-12.9
ROCM_PATH   ?= /opt/rocm
CXX         := g++

ROOT_CFLAGS := $(shell root-config --cflags)
ROOT_LIBS   := $(shell root-config --libs)
XML2_CFLAGS := $(shell xml2-config --cflags)
XML2_LIBS   := $(shell xml2-config --libs)

ifeq ($(GPU_BACKEND),cuda)
  DEVCC := $(CUDA_PATH)/bin/nvcc
  # Double-precision atomicAdd needs sm_60 or newer. Architectures this toolkit
  # does not support are dropped (CUDA 13 removed sm_70; CUDA < 11.8 lacks
  # sm_89/sm_90). PTX for the newest one keeps future GPUs working via JIT.
  # CUDA_ARCHS=native builds only for the GPU(s) in this machine.
  CUDA_ARCHS    ?= 70 75 80 86 89 90
  ifeq ($(strip $(CUDA_ARCHS)),native)
    CUDA_BUILD  := native
    GENCODE     := -arch=native
  else
    CUDA_KNOWN    := $(patsubst compute_%,%,$(shell $(DEVCC) --list-gpu-arch 2>/dev/null))
    CUDA_BUILD    := $(filter $(CUDA_KNOWN),$(CUDA_ARCHS))
    CUDA_PTX_ARCH ?= $(lastword $(CUDA_BUILD))
    GENCODE       := $(foreach a,$(CUDA_BUILD),-gencode arch=compute_$(a),code=sm_$(a)) \
                     -gencode arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)
  endif
  DEVFLAGS      := -O3 -std=c++17 -Xcompiler -fPIC $(GENCODE) -Wno-deprecated-gpu-targets --threads 0
  GPU_INCLUDES  := -I$(CUDA_PATH)/include
  GPU_DEFINES   :=
  GPU_LIBS      := -L$(CUDA_PATH)/lib64 -lcudart -Wl,-rpath,$(CUDA_PATH)/lib64
  ifeq ($(filter clean,$(MAKECMDGOALS)),)
    ifeq ($(strip $(CUDA_BUILD)),)
      $(error No usable CUDA architecture: check CUDA_PATH=$(CUDA_PATH) and CUDA_ARCHS=$(CUDA_ARCHS))
    endif
  endif
else ifeq ($(GPU_BACKEND),hip)
  DEVCC         := $(ROCM_PATH)/bin/hipcc
  HIP_ARCHS     ?= gfx90a gfx942 gfx1100
  DEVFLAGS      := -x hip -O3 -std=c++17 -fPIC $(foreach a,$(HIP_ARCHS),--offload-arch=$(a))
  GPU_INCLUDES  := -I$(ROCM_PATH)/include
  # Host files are compiled by g++ against the HIP runtime API.
  GPU_DEFINES   := -DUSE_HIP -D__HIP_PLATFORM_AMD__
  GPU_LIBS      := -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
else
  $(error GPU_BACKEND must be cuda or hip)
endif

INCLUDES    := -Iinclude $(GPU_INCLUDES) $(XML2_CFLAGS)
CXXFLAGS    := -O3 -std=c++17 -fPIC -fopenmp -Wall $(ROOT_CFLAGS) $(GPU_DEFINES) $(INCLUDES)
NVCCFLAGS   := $(DEVFLAGS) $(INCLUDES)

OBJ_DIR     := build
LIB_DIR     := lib

OBJS        := $(OBJ_DIR)/gpu_spline_kernel.o \
               $(OBJ_DIR)/gpu_flux_kernel.o \
               $(OBJ_DIR)/gpu_hadron_tensor_kernel.o \
               $(OBJ_DIR)/gpu_qel_integrator.o \
               $(OBJ_DIR)/spline_loader.o \
               $(OBJ_DIR)/gpu_spline_engine.o \
               $(OBJ_DIR)/gpu_spline_c_api.o \
               $(OBJ_DIR)/gpu_flux_preselector.o \
               $(OBJ_DIR)/hadron_tensor_loader.o \
               $(OBJ_DIR)/gpu_status.o

TARGET_LIB  := $(LIB_DIR)/libgpuspline.so

.PHONY: all lib clean dirs

all: dirs $(TARGET_LIB)

lib: dirs $(TARGET_LIB)

# The portability and physics helpers are inline; rebuild their consumers.
$(OBJS): $(wildcard include/gpu_spline/*.h)
$(OBJS): | dirs

dirs:
	@mkdir -p $(OBJ_DIR) $(LIB_DIR)

# Device kernel compilation (nvcc, or hipcc for GPU_BACKEND=hip)
$(OBJ_DIR)/gpu_spline_kernel.o: src/gpu_spline_kernel.cu
	$(DEVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_flux_kernel.o: src/gpu_flux_kernel.cu
	$(DEVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_hadron_tensor_kernel.o: src/gpu_hadron_tensor_kernel.cu
	$(DEVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/gpu_qel_integrator.o: src/gpu_qel_integrator.cu
	$(DEVCC) $(NVCCFLAGS) -c $< -o $@

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

$(OBJ_DIR)/gpu_status.o: src/gpu_status.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Shared Library
$(TARGET_LIB): $(OBJS)
	$(CXX) -shared -o $@ $^ $(GPU_LIBS) $(ROOT_LIBS) $(XML2_LIBS) -lz -fopenmp

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
