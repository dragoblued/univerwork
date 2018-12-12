#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <iostream>
#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define m 6
#define n 5

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int main(void) {
  thrust::host_vector<int> A(m * n);
  thrust::host_vector<int> A2(m);
  thrust::host_vector<int> X(n);
  thrust::host_vector<int> Y(m);
  int i, j;
  int ind = 11;
  float Res = 0.0f;
  float* RV;
  RV = (float*)malloc(m * sizeof(RV));
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      A[IDX2C(i, j, m)] = (float)ind++;
    }
  }
  printf("Matrix:\n");
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      std::cout << A[IDX2C(i, j, m)] << " " << std::ends;
    }
    printf("\n");
  }
  for (i = 0; i < n; i++) {
    X[i] = 1.0f;
  }
  printf("Vector:\n");
  for (i = 0; i < n; i++) {
    std::cout << X[i] << " " << std::ends;
  }
  printf("\n");
  for (i = 0; i < m; i++) {
    Y[i] = 0.0f;
  }
  double t = wtime();
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      A2[j] = A[IDX2C(i, j, m)];
    }
    thrust::transform(A2.begin(), A2.end(), X.begin(), Y.begin(), thrust::multiplies<float>());
    for (j = 0; j < n; j++) {
      Res = thrust::reduce(Y.begin(), Y.end());
    }
    RV[i] = Res;
    Res = 0.0f;
  }
  t = wtime() - t;
  printf("Result vector:\n");
  for (i = 0; i < m; i++) {
    printf("%4.0f", RV[i]);
  }
  printf("\n");
  //printf("Time:%.9f\n", t);
  free(RV);
}
