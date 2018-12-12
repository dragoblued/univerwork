#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 100000

void swap (double *a, int i, int j) {
	double z = a[i];
	a[i] = a[j];
	a [j] = z;
}


double getrand()
{
	return (double)rand()/RAND_MAX;
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int partition(double *a, int low, int high)
{
	double pivot = a[high];
	int j = low;
	for (int i = low; i <= high - 1; i++) {
		if (a[i] <= pivot) {
			swap(a, i, j);
			j++;		
		}	
	}
	if (a[j] > a[high])swap(a, j, high);
	return j;
}

void quick_sort(double *a, int low, int high)
{
	int p;
	if (low < high) {
		p = partition(a, low, high);
		quick_sort(a, low, p - 1);
		quick_sort(a, p+1, high);	
	}	
}

int main(int argc, char **argv)
{
	double *a = malloc(sizeof(*a) * N);
	for (int i = 0; i < N; i++) {
		a[i] = getrand();
		printf("%.2f ", a[i]);
	}
	printf("\n end\n");
	quick_sort(a, 0, N - 1);
	for (int i = 0; i < N; i++) {
		printf("%.2f ", a[i]);
	}
	free(a);
	return 0;
}
