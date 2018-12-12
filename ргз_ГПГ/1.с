#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define m 6
#define n 5
__global__ void gSum(float* a, float* b, float* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  c[i] = 0;
  for (j = 0; j < n; j++) {
    c[i] += a[IDX2C(i, j, m)] * b[j];
  }
}

int main() {
  float *a, *x, *y;
  int i, j;
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
  float *d_a, *d_x, *d_y;
  cudaMalloc((void**)&d_a, m * n *sizeof(*a));
  cudaMalloc((void**)&d_x, n * sizeof(*x));
  cudaMalloc((void**)&d_y, m * sizeof(*y));
  cudaMemcpy(d_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  dim3 threads(m, 1, 1);
  dim3 grid(1, 1, 1);
  gSum<<<grid, threads>>>(d_a, d_x, d_y);
  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
  printf("Result vector:\n");
  for (i = 0; i < m; i++) {
    printf("%5.0f ", y[i]);
  }
  printf("\n");
  free(a);
  free(x);
  free(y);
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}
