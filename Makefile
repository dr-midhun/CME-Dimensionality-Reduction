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
all: $(BUILD)/cme_cli $(BUILD)/test_small $(BUILD)/test_exhaustive

$(BUILD)/cme_cli: apps/cme_cli.cpp $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LDLIBS)
$(BUILD)/test_small: tests/test_small.cpp $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LDLIBS)
$(BUILD)/test_exhaustive: tests/test_exhaustive.cpp $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LDLIBS)
$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/test_small $(BUILD)/test_exhaustive
	./$(BUILD)/test_small
	./$(BUILD)/test_exhaustive

# Runs the GPU code path against a CPU mock of cuBLAS/cuSPARSE (logic check only, no GPU needed).
test-gpu-logic: tests/test_small.cpp tests/test_exhaustive.cpp $(SRC) src/alg5_eigenbasis_cuda.cpp tests/mock_cuda/mock.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DCME_WITH_CUDA -Itests/mock_cuda -Iinclude tests/test_small.cpp $(SRC) src/alg5_eigenbasis_cuda.cpp tests/mock_cuda/mock.cpp -o $(BUILD)/test_gpu_logic $(LDLIBS)
	$(CXX) $(CXXFLAGS) -DCME_WITH_CUDA -Itests/mock_cuda -Iinclude tests/test_exhaustive.cpp $(SRC) src/alg5_eigenbasis_cuda.cpp tests/mock_cuda/mock.cpp -o $(BUILD)/test_gpu_exhaustive $(LDLIBS)
	./$(BUILD)/test_gpu_logic gpu
	./$(BUILD)/test_gpu_exhaustive gpu
clean:
	rm -rf $(BUILD)

# ==============================================================================
# Dedicated targets for each Thesis Algorithm (1 through 12)
# ==============================================================================
NETWORK ?= examples/michaelis_menten.net
OUT_DIR ?= out

run-alg1: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 1 --out $(OUT_DIR)/alg1

run-alg2: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 2 --out $(OUT_DIR)/alg2

run-alg3: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 3 --out $(OUT_DIR)/alg3

run-alg4: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 4 --d 4 --solve-time 0.1 --out $(OUT_DIR)/alg4

run-alg5: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 5 --d 4 --out $(OUT_DIR)/alg5

run-alg6: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 6 --tf 0.1 --eps 1e-4 --out $(OUT_DIR)/alg6

run-alg7: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 7 --tf 0.1 --num-steps 5 --eps 1e-4 --out $(OUT_DIR)/alg7

run-alg8: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 8 --tf 0.1 --eps 1e-4 --out $(OUT_DIR)/alg8

run-alg9: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 9 --d 4 --tf 0.5 --eps 1e-4 --t-eps 0.01 --out $(OUT_DIR)/alg9

run-alg10: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 10 --d 4 --eps 1e-4 --t-eps 0.01 --out $(OUT_DIR)/alg10

run-alg11: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 11 --d 4 --eps 1e-4 --t-eps 0.01 --out $(OUT_DIR)/alg11

run-alg12: $(BUILD)/cme_cli
	@mkdir -p $(OUT_DIR)
	./$(BUILD)/cme_cli $(NETWORK) --alg 12 --d 4 --tf 0.5 --eps 1e-4 --t-eps 0.01 --out $(OUT_DIR)/alg12

run-all-algs: run-alg1 run-alg2 run-alg3 run-alg4 run-alg5 run-alg6 run-alg7 run-alg8 run-alg9 run-alg10 run-alg11 run-alg12

init:
	@./utils/setup_prereqs.sh

.PHONY: all test test-gpu-logic clean init run-all-algs run-alg1 run-alg2 run-alg3 run-alg4 run-alg5 run-alg6 run-alg7 run-alg8 run-alg9 run-alg10 run-alg11 run-alg12
