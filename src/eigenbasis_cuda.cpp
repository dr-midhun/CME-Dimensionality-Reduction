// GPU version of thesis Algorithm 5 (see eigenbasis_cpu.cpp for the algorithmic notes).
// Plain C++ host code against the CUDA runtime, cuBLAS and cuSPARSE: no custom kernels, so it
// builds with any C++ compiler that has the CUDA headers (nvcc is not required).
//
// Layout: V, W, T are n x d ROW-major device matrices (leading dimension d).  For cuBLAS the same
// memory is the column-major d x n matrix V^T, which turns "V * S" into  (S^T) (V^T).
// Memory: matrix (12 B/nnz + 4 B/row) + 3 n x d doubles.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "cme/eigenbasis.hpp"
#include "cme/small_dense.hpp"

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x); } while (0)
#define BLAS_OK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cuBLAS error ") + std::to_string((int)s_) + " at " #x); } while (0)
#define SPARSE_OK(x) do { cusparseStatus_t s_ = (x); if (s_ != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string("cuSPARSE: ") + cusparseGetErrorString(s_) + " at " #x); } while (0)

namespace cme {
namespace {

constexpr int MAXD = 64;

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes = 0) { if (bytes) CUDA_OK(cudaMalloc(&p, bytes)); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
};

}  // namespace

EigenResult eigenbasis_cuda(const Csr& D, double sigma, const EigenOptions& opt) {
    const int d = opt.d;
    const int n = D.n;
    if (d < 1 || d > MAXD) throw std::runtime_error("d must be in [1, 64]");
    const size_t nd = static_cast<size_t>(n) * d;
    if (nd >= 2147483647ull) throw std::runtime_error("n*d must be < 2^31 for the 32-bit cuBLAS calls used here");
    const bool slow = opt.mode == Mode::Slow;
    const bool cheby = opt.cutoff > 0.0;
    const double ka = 1.0 / sigma, kb = slow ? 1.0 : 0.0;

    cublasHandle_t blas;
    cusparseHandle_t sparse;
    BLAS_OK(cublasCreate(&blas));
    SPARSE_OK(cusparseCreate(&sparse));

    // --- matrix on the device
    DevBuf d_ptr((n + 1) * sizeof(int32_t)), d_col(D.col.size() * sizeof(int32_t)), d_val(D.val.size() * sizeof(double));
    CUDA_OK(cudaMemcpy(d_ptr.p, D.row_ptr.data(), (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_col.p, D.col.data(), D.col.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_val.p, D.val.data(), D.val.size() * sizeof(double), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t matA;
    SPARSE_OK(cusparseCreateCsr(&matA, n, n, D.nnz(), d_ptr.p, d_col.p, d_val.p, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // --- dense work matrices
    DevBuf bufV(nd * 8), bufW(nd * 8), bufT(nd * 8), bufU(static_cast<size_t>(n) * 8), bufS(d * d * 8), bufw(d * 8);
    double* V = bufV.as<double>();
    double* W = bufW.as<double>();
    double* T = bufT.as<double>();
    double* U = bufU.as<double>();
    double* dS = bufS.as<double>();
    double* dw = bufw.as<double>();
    CUDA_OK(cudaMemset(W, 0, nd * 8));
    CUDA_OK(cudaMemset(T, 0, nd * 8));
    {
        std::vector<double> h(nd);
        std::mt19937_64 gen(opt.seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& x : h) x = dist(gen);
        CUDA_OK(cudaMemcpy(V, h.data(), nd * 8, cudaMemcpyHostToDevice));
    }

    // SpMM workspace (same shapes for every call).
    auto dn = [&](double* p) {
        cusparseDnMatDescr_t m;
        SPARSE_OK(cusparseCreateDnMat(&m, n, d, d, p, CUDA_R_64F, CUSPARSE_ORDER_ROW));
        return m;
    };
    DevBuf spbuf;
    {
        cusparseDnMatDescr_t x = dn(V), y = dn(W);
        const double one = 1.0, zero = 0.0;
        size_t bytes = 0;
        SPARSE_OK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &one, matA, x, &zero, y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
        if (bytes) { CUDA_OK(cudaMalloc(&spbuf.p, bytes)); }
        cusparseDestroyDnMat(x);
        cusparseDestroyDnMat(y);
    }

    EigenResult res;
    res.n = n;
    res.d = d;

    // out = a * D X + b * X + c * Z      (Z, if used, must be `out` itself)
    auto spmm = [&](const double* X, double* out, double a, double b, double c) {
        cusparseDnMatDescr_t x = dn(const_cast<double*>(X)), y = dn(out);
        const double beta = c;  // beta * C with C = out (== Z when c != 0)
        SPARSE_OK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &a, matA, x,
                               &beta, y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, spbuf.p));
        cusparseDestroyDnMat(x);
        cusparseDestroyDnMat(y);
        if (b != 0.0) BLAS_OK(cublasDaxpy(blas, static_cast<int>(nd), &b, X, 1, out, 1));
        ++res.spmm_count;
    };
    auto maxabs = [&](const double* x, int count) {
        int idx = 1;
        double v = 0.0;
        BLAS_OK(cublasIdamax(blas, count, x, 1, &idx));
        CUDA_OK(cudaMemcpy(&v, x + (idx - 1), 8, cudaMemcpyDeviceToHost));
        return std::fabs(v);
    };
    auto read_row = [&](const double* M, int64_t row, double* host) {
        CUDA_OK(cudaMemcpy(host, M + row * d, d * 8, cudaMemcpyDeviceToHost));
    };

    std::vector<int64_t> piv, piv_prev;
    bool frozen = false;
    int stable = 0;

    // Gauss-Jordan on columns of V (row-major): afterwards V[piv[j],:] = e_j.
    auto pivot_search = [&]() {
        piv.assign(d, 0);
        double w[MAXD];
        for (int j = 0; j < d; ++j) {
            int idx = 1;
            BLAS_OK(cublasIdamax(blas, n, V + j, d, &idx));
            const int64_t p = idx - 1;
            read_row(V, p, w);
            const double pv = w[j];
            if (std::fabs(pv) < 1e-300) throw std::runtime_error("basis lost rank during pivoting");
            piv[j] = p;
            const double inv = 1.0 / pv;
            BLAS_OK(cublasDscal(blas, n, &inv, V + j, d));
            BLAS_OK(cublasDcopy(blas, n, V + j, d, U, 1));   // scaled column j
            w[j] = 0.0;
            CUDA_OK(cudaMemcpy(dw, w, d * 8, cudaMemcpyHostToDevice));
            const double m1 = -1.0;
            // V^T (d x n, col-major) += -w * u^T
            BLAS_OK(cublasDger(blas, d, n, &m1, dw, 1, U, 1, V, d));
        }
    };
    // V <- V * inv(V[piv,:]); returns max|V|.
    auto pivot_fixed = [&]() {
        std::vector<double> S(static_cast<size_t>(d) * d);
        for (int j = 0; j < d; ++j) read_row(V, piv[j], &S[j * d]);
        dense_inverse(d, S.data());
        CUDA_OK(cudaMemcpy(dS, S.data(), S.size() * 8, cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0;
        // W^T = (S^-1)^T V^T   (dS holds S^-1 row-major == (S^-1)^T column-major)
        BLAS_OK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, d, n, d, &one, dS, d, V, d, &zero, W, d));
        std::swap(V, W);
        return maxabs(V, static_cast<int>(nd));
    };
    auto normalise = [&]() {
        if (!frozen) {
            pivot_search();
            stable = (piv == piv_prev) ? stable + 1 : 0;
            piv_prev = piv;
            if (stable >= 2) frozen = true;
        } else if (pivot_fixed() > 1e3) {
            frozen = false;
            stable = 0;
            pivot_search();
            piv_prev = piv;
        }
    };

    std::vector<double> Kred(static_cast<size_t>(d) * d);
    // W = K V; Kred = W[piv,:]; returns max|V Kred - W| (T is scratch).
    auto residual = [&]() {
        spmm(V, W, ka, kb, 0.0);
        for (int j = 0; j < d; ++j) read_row(W, piv[j], &Kred[j * d]);
        CUDA_OK(cudaMemcpy(dS, Kred.data(), Kred.size() * 8, cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0, m1 = -1.0;
        BLAS_OK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, d, n, d, &one, dS, d, V, d, &zero, T, d));  // T = V Kred
        BLAS_OK(cublasDaxpy(blas, static_cast<int>(nd), &m1, W, 1, T, 1));                              // T -= W
        return maxabs(T, static_cast<int>(nd));
    };

    // Chebyshev parameters (identical to the CPU path).
    double lo = 0, hi = 0, r0 = 0, e = 1, ctr = 0, s1 = 1;
    if (cheby) {
        if (slow) { lo = -sigma; hi = -opt.cutoff; r0 = 0.0; }
        else      { lo = -opt.cutoff; hi = 0.0; r0 = -sigma; }
        e = 0.5 * (hi - lo);
        ctr = 0.5 * (hi + lo);
        s1 = e / (r0 - ctr);
    }

    CUDA_OK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    int it = 0;
    for (; it < opt.max_iter; ++it) {
        normalise();
        if (cheby || it % std::max(1, opt.check_every) == 0) {
            res.residual = residual();
            if (opt.verbose) std::fprintf(stderr, "  iter %6d  residual %.3e%s\n", it, res.residual, frozen ? "  (pivots frozen)" : "");
            if (res.residual < opt.tol) { res.converged = true; break; }
        }
        if (cheby) {
            double* Y0 = W;  // previous
            double* Y1 = T;  // current
            double sk = s1;
            spmm(V, Y1, s1 / e, -s1 * ctr / e, 0.0);
            CUDA_OK(cudaMemcpy(Y0, V, nd * 8, cudaMemcpyDeviceToDevice));
            for (int k = 1; k < opt.degree; ++k) {
                const double sn = 1.0 / (2.0 / s1 - sk);
                spmm(Y1, Y0, 2.0 * sn / e, -2.0 * sn * ctr / e, -sn * sk);  // in place on Y0 (holds Y_{k-1})
                std::swap(Y0, Y1);
                sk = sn;
            }
            // newest = Y1 becomes V; the other two buffers become W and T
            double* old_v = V;
            V = Y1;
            W = old_v;
            T = Y0;
        } else {
            for (int s = 0; s < opt.inner; ++s) {
                spmm(V, W, ka, kb, 0.0);
                std::swap(V, W);
            }
        }
    }
    if (!res.converged) {
        normalise();
        res.residual = residual();
        res.converged = res.residual < opt.tol;
    }
    res.iterations = it;
    CUDA_OK(cudaDeviceSynchronize());
    res.seconds_iterate = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    res.basis.resize(nd);
    CUDA_OK(cudaMemcpy(res.basis.data(), V, nd * 8, cudaMemcpyDeviceToHost));
    res.d_reduced = Kred;
    res.pivots = piv;

    cusparseDestroySpMat(matA);
    cusparseDestroy(sparse);
    cublasDestroy(blas);
    return res;
}

}  // namespace cme
