#include "conv_util.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[i*lda + k];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}
void gemm_nt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i*lda + k] * B[j*ldb + k];
			}
			C[i*ldc + j] += sum;
		}
	}
}
void gemm_tn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[k*lda + i];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}
void gemm_tt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i*ldc + j] += sum;
		}
	}
}
#if __ANDROID__ == 1
#include <include/cblas.h>
#elif __APPLE__ == 1
#include <Accelerate/Accelerate.h>
#elif CPU_GEMM == 1
#else
#include <mkl.h>
#endif
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc)
{
#if CPU_GEMM == 1
	for (int i = 0; i < N * M; ++i)
		C[i] *= BETA;
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB)
		gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB)
		gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else
		gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#else
	if (!TA && !TB)
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
	else if (TA && !TB)
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
	else if (!TA && TB)
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
	else
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
#endif
}
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc)
{
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
//	printf("%d,%d,%d,%u\n", M, N, N, (unsigned int)M * (unsigned int)N * (unsigned int)K);
}