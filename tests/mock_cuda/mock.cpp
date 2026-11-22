// CPU implementation of the few CUDA / cuBLAS / cuSPARSE calls used by src/eigenbasis_cuda.cpp,
// following the documented (column-major BLAS, CSR SpMM) semantics.  Lets CI exercise the GPU code
// path's logic on machines without a GPU:   make test-gpu-logic
// It validates layout conventions and algorithm flow, NOT CUDA runtime behaviour or performance.
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusparse.h"

struct cublasContext {};
struct cusparseContext {};
struct cusparseSpMatDescr { int64_t rows, cols, nnz; const int* ptr; const int* col; const double* val; };
struct cusparseDnMatDescr { int64_t rows, cols, ld; double* p; cusparseOrder_t order; };

const char* cudaGetErrorString(cudaError_t) { return "mock"; }
cudaError_t cudaMalloc(void** p, size_t n) { *p = std::malloc(n); return *p ? 0 : 1; }
cudaError_t cudaFree(void* p) { std::free(p); return 0; }
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return 0; }
cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return 0; }
cudaError_t cudaDeviceSynchronize() { return 0; }

cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = new cublasContext; return 0; }
cublasStatus_t cublasDestroy(cublasHandle_t h) { delete h; return 0; }
cublasStatus_t cublasDaxpy(cublasHandle_t, int n, const double* a, const double* x, int incx, double* y, int incy) {
    for (int i = 0; i < n; ++i) y[(int64_t)i * incy] += *a * x[(int64_t)i * incx];
    return 0;
}
cublasStatus_t cublasIdamax(cublasHandle_t, int n, const double* x, int incx, int* r) {
    int best = 0; double bv = -1;
    for (int i = 0; i < n; ++i) { double v = std::fabs(x[(int64_t)i * incx]); if (v > bv) { bv = v; best = i; } }
    *r = best + 1;  // 1-based, like cuBLAS
    return 0;
}
cublasStatus_t cublasDscal(cublasHandle_t, int n, const double* a, double* x, int incx) {
    for (int i = 0; i < n; ++i) x[(int64_t)i * incx] *= *a;
    return 0;
}
cublasStatus_t cublasDcopy(cublasHandle_t, int n, const double* x, int incx, double* y, int incy) {
    for (int i = 0; i < n; ++i) y[(int64_t)i * incy] = x[(int64_t)i * incx];
    return 0;
}
// A (m x n, column-major, lda) += alpha * x * y^T
cublasStatus_t cublasDger(cublasHandle_t, int m, int n, const double* alpha, const double* x, int incx,
                          const double* y, int incy, double* A, int lda) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) A[(int64_t)j * lda + i] += *alpha * x[(int64_t)i * incx] * y[(int64_t)j * incy];
    return 0;
}
// C (m x n) = alpha op(A) (m x k) op(B) (k x n) + beta C, column-major
cublasStatus_t cublasDgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           const double* alpha, const double* A, int lda, const double* B, int ldb,
                           const double* beta, double* C, int ldc) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0;
            for (int p = 0; p < k; ++p) {
                double a = (ta == CUBLAS_OP_N) ? A[(int64_t)p * lda + i] : A[(int64_t)i * lda + p];
                double b = (tb == CUBLAS_OP_N) ? B[(int64_t)j * ldb + p] : B[(int64_t)p * ldb + j];
                acc += a * b;
            }
            double& c = C[(int64_t)j * ldc + i];
            c = *alpha * acc + (*beta == 0.0 ? 0.0 : *beta * c);
        }
    return 0;
}

const char* cusparseGetErrorString(cusparseStatus_t) { return "mock"; }
cusparseStatus_t cusparseCreate(cusparseHandle_t* h) { *h = new cusparseContext; return 0; }
cusparseStatus_t cusparseDestroy(cusparseHandle_t h) { delete h; return 0; }
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* d, int64_t r, int64_t c, int64_t nnz, void* ptr, void* col,
                                   void* val, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) {
    *d = new cusparseSpMatDescr{r, c, nnz, (const int*)ptr, (const int*)col, (const double*)val};
    return 0;
}
cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* d, int64_t r, int64_t c, int64_t ld, void* p, cudaDataType,
                                     cusparseOrder_t o) {
    *d = new cusparseDnMatDescr{r, c, ld, (double*)p, o};
    return 0;
}
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t d) { delete d; return 0; }
cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t d) { delete d; return 0; }
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*,
                                         cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void*, cusparseDnMatDescr_t,
                                         cudaDataType, cusparseSpMMAlg_t, size_t* bytes) {
    *bytes = 0;
    return 0;
}
// C = alpha * A * B + beta * C   (row-major dense B, C)
cusparseStatus_t cusparseSpMM(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void* alpha,
                              cusparseSpMatDescr_t A, cusparseDnMatDescr_t B, const void* beta, cusparseDnMatDescr_t C,
                              cudaDataType, cusparseSpMMAlg_t, void*) {
    const double a = *(const double*)alpha, b = *(const double*)beta;
    const int d = (int)B->cols;
    for (int64_t i = 0; i < A->rows; ++i)
        for (int q = 0; q < d; ++q) {
            double acc = 0;
            for (int e = A->ptr[i]; e < A->ptr[i + 1]; ++e) acc += A->val[e] * B->p[(int64_t)A->col[e] * B->ld + q];
            double& c = C->p[i * C->ld + q];
            c = a * acc + (b == 0.0 ? 0.0 : b * c);
        }
    return 0;
}
