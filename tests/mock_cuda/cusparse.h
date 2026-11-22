#pragma once
#include "cuda_runtime.h"
struct cusparseContext; typedef cusparseContext* cusparseHandle_t;
struct cusparseSpMatDescr; typedef cusparseSpMatDescr* cusparseSpMatDescr_t;
struct cusparseDnMatDescr; typedef cusparseDnMatDescr* cusparseDnMatDescr_t;
typedef int cusparseStatus_t; enum { CUSPARSE_STATUS_SUCCESS = 0 };
enum cusparseIndexType_t { CUSPARSE_INDEX_32I = 2 }; enum cusparseIndexBase_t { CUSPARSE_INDEX_BASE_ZERO };
enum cusparseOrder_t { CUSPARSE_ORDER_ROW = 1 }; enum cusparseOperation_t { CUSPARSE_OPERATION_NON_TRANSPOSE };
enum cusparseSpMMAlg_t { CUSPARSE_SPMM_ALG_DEFAULT };
const char* cusparseGetErrorString(cusparseStatus_t);
cusparseStatus_t cusparseCreate(cusparseHandle_t*); cusparseStatus_t cusparseDestroy(cusparseHandle_t);
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t*,int64_t,int64_t,int64_t,void*,void*,void*,cusparseIndexType_t,cusparseIndexType_t,cusparseIndexBase_t,cudaDataType);
cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t*,int64_t,int64_t,int64_t,void*,cudaDataType,cusparseOrder_t);
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t); cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t);
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t,cusparseOperation_t,cusparseOperation_t,const void*,cusparseSpMatDescr_t,cusparseDnMatDescr_t,const void*,cusparseDnMatDescr_t,cudaDataType,cusparseSpMMAlg_t,size_t*);
cusparseStatus_t cusparseSpMM(cusparseHandle_t,cusparseOperation_t,cusparseOperation_t,const void*,cusparseSpMatDescr_t,cusparseDnMatDescr_t,const void*,cusparseDnMatDescr_t,cudaDataType,cusparseSpMMAlg_t,void*);
