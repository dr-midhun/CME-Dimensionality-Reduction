#pragma once
#include "cuda_runtime.h"
struct cublasContext; typedef cublasContext* cublasHandle_t;
typedef int cublasStatus_t; enum { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
cublasStatus_t cublasCreate(cublasHandle_t*); cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasDaxpy(cublasHandle_t,int,const double*,const double*,int,double*,int);
cublasStatus_t cublasIdamax(cublasHandle_t,int,const double*,int,int*);
cublasStatus_t cublasDscal(cublasHandle_t,int,const double*,double*,int);
cublasStatus_t cublasDcopy(cublasHandle_t,int,const double*,int,double*,int);
cublasStatus_t cublasDger(cublasHandle_t,int,int,const double*,const double*,int,const double*,int,double*,int);
cublasStatus_t cublasDgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
