# Multi-stage Dockerfile for Dimensionality Reduction of the Chemical Master Equation (CME)
# Provides a self-contained, reproducible Ubuntu Linux environment with OpenMP and LAPACK.

# ==============================================================================
# Stage 1: Build environment
# ==============================================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install C++ compiler toolchain, OpenMP, LAPACK, BLAS, and build systems
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    g++ \
    libblas-dev \
    liblapack-dev \
    libomp-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source tree into container
COPY include/ include/
COPY src/ src/
COPY apps/ apps/
COPY tests/ tests/
COPY examples/ examples/
COPY utils/ utils/
COPY Makefile CMakeLists.txt ./

# Compile library and executables with full OpenMP and LAPACK optimization
RUN make -j$(nproc) LAPACK=1

# Execute complete test suite inside builder stage to verify mathematical correctness
RUN ./build/test_small
RUN ./build/test_exhaustive
RUN make test-gpu-logic

# ==============================================================================
# Stage 2: Lightweight runtime image
# ==============================================================================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install OpenMP runtime, LAPACK runtime, BLAS runtime, and Python for SciPy/NumPy verification
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libblas3 \
    libgomp1 \
    liblapack3 \
    python3 \
    python3-numpy \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /cme

# Copy compiled binaries from builder stage
COPY --from=builder /build/build/cme_cli /usr/local/bin/cme_cli
COPY --from=builder /build/build/test_small /usr/local/bin/test_small
COPY --from=builder /build/build/test_exhaustive /usr/local/bin/test_exhaustive
COPY --from=builder /build/build/test_gpu_logic /usr/local/bin/test_gpu_logic
COPY --from=builder /build/build/test_gpu_exhaustive /usr/local/bin/test_gpu_exhaustive

# Copy examples and verification scripts
COPY examples/ /cme/examples/
COPY utils/verify.py /cme/utils/verify.py

# Default volume for mounting user datasets and saving output files
VOLUME ["/cme/data"]

# Default entrypoint runs the CME CLI
ENTRYPOINT ["cme_cli"]
CMD ["--help"]
