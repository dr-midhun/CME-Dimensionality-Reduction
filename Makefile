# CPU build:   make                (auto-detects Linux / macOS with OpenMP)
# + LAPACK:    make LAPACK=1       (eigenvalues of the d x d reduced matrix; LAPACK_LIBS=-llapack)
# + GPU:       make GPU=1          (CUDA toolkit: cudart, cublas, cusparse; CUDA_HOME=/usr/local/cuda)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXX ?= clang++
else
  CXX ?= g++
endif

CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
INC      := -Iinclude
SRC      := src/network.cpp src/csr_util.cpp \
            src/alg1_cme.cpp \
            src/alg2_state_space.cpp \
            src/alg3_transition.cpp \
            src/alg4_reduction.cpp \
            src/alg5_eigenbasis.cpp src/alg5_eigenbasis_cpu.cpp \
            src/alg6_7_8_fsp.cpp \
            src/alg9_10_11_12_preconditioned_fsp.cpp
LDLIBS   :=

ifeq ($(UNAME_S),Darwin)
  # Apple Clang requires -Xpreprocessor -fopenmp and linking against libomp (e.g. from Homebrew)
  BREW_PREFIX ?= $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
  CXXFLAGS += -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
  LDLIBS   += -L$(BREW_PREFIX)/opt/libomp/lib -lomp
else
  # Linux / GCC / Clang
  CXXFLAGS += -fopenmp
endif

ifeq ($(LAPACK),1)
  LAPACK_LIBS ?= -llapack
  CXXFLAGS += -DCME_WITH_LAPACK
  LDLIBS   += $(LAPACK_LIBS)
endif
ifeq ($(GPU),1)
  CUDA_HOME ?= /usr/local/cuda
  CXXFLAGS += -DCME_WITH_CUDA
  INC      += -I$(CUDA_HOME)/include
  SRC      += src/alg5_eigenbasis_cuda.cpp
  LDLIBS   += -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse
endif

BUILD := build
all: $(BUILD)/cme_cli $(BUILD)/test_small

$(BUILD)/cme_cli: apps/cme_cli.cpp $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LDLIBS)
$(BUILD)/test_small: tests/test_small.cpp $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LDLIBS)
$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/test_small
	./$(BUILD)/test_small

# Runs the GPU code path against a CPU mock of cuBLAS/cuSPARSE (logic check only, no GPU needed).
test-gpu-logic: tests/test_small.cpp $(SRC) src/alg5_eigenbasis_cuda.cpp tests/mock_cuda/mock.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DCME_WITH_CUDA -Itests/mock_cuda -Iinclude $^ -o $(BUILD)/test_gpu_logic $(LDLIBS)
	./$(BUILD)/test_gpu_logic gpu
clean:
	rm -rf $(BUILD)

init:
	@./utils/setup_prereqs.sh

.PHONY: all test test-gpu-logic clean init
