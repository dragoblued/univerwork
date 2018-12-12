#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define m 6
#define n 5

int main(void) {
  cublasHandle_t handle;
  int i, j;
  float* a;
  float* x;
  float* y;
  a = (float*)malloc(m * n * sizeof(float));
  x = (float*)malloc(n * sizeof(float));
  y = (float*)malloc(m * sizeof(float));
  int ind = 11;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      a[IDX2C(i, j, m)] = (float)ind++;
    }
  }
  printf("Matrix:\n");
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%4.0f", a[IDX2C(i, j, m)]);
    }
    printf("\n");
  }
  for (i = 0; i < n; i++) {
    x[i] = 1.0f;
  }
  printf("Vector:\n");
  for (i = 0; i < n; i++) {
    printf("%4.0f", x[i]);
  }
  printf("\n");
  for (i = 0; i < m; i++) {
    y[i] = 0.0f;
  }
  float* d_a;
  float* d_x;
  float* d_y;
  cudaMalloc((void**) &d_a, m * n * sizeof(float));
  cudaMalloc((void**) &d_x, n * sizeof(float));
  cudaMalloc((void**) &d_y, m * sizeof(float));
  cublasCreate(&handle);
  cublasSetMatrix(m, n, sizeof(*a), a, m, d_a, m);
  cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
  cublasSetVector(m, sizeof(*y), y, 1, d_y, 1);
  float al = 1.0f;
  float bet = 1.0f;
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &al, d_a, m, d_x, 1, &bet, d_y, 1);
  cublasGetVector(m, sizeof(*y), d_y, 1, y, 1);
  printf("Result vector:\n");
  for (j = 0; j < m; j++) {
    printf("%5.0f", y[j]);
  }
  printf("\n");
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  cublasDestroy(handle);
  return 0;
}
