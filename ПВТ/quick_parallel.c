#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#define N 100000

void swap(double *a, int i, int j)
{
	double z = a[i];
	a[i] = a[j];
	a[j] = z;
}

double getrand()
{
	return (double)rand() / RAND_MAX;
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
	if (a[j] > a[high]) swap(a, j, high);
	return j;
}

void print_mas(double *a, int i, int j) {
	for (int  k = i; k < j; k++) {
		printf(" %.2f ", a[k]);	
	}	
	printf("\n");
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

void quick_sort_parallel(double *a, int low, int high)
{
	int rank, commsize,  p;
	MPI_Status Status;
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
		int mas_begin[commsize];
		int mas_end[commsize];
	if (rank == 0) {

		p = partition(a, 0, N - 1);
		mas_begin[0] = 0;
		mas_end[0] = p - 1;	
		mas_begin[1] = p + 1;
		mas_end[1] = N - 1;
		for (int i = 2; i < commsize; i++) {
				if(i % 2 == 0) {
					p = partition(a, mas_begin[0], mas_end[0]);
					mas_begin[i] = p;
					mas_end[i] = mas_end[0];
					mas_begin[0] = 0;
					mas_end[0] = p - 1;
					//printf("\nmas %d %d p %d\n", mas_begin[i], mas_end[i], p);	
					//printf("\nmas %d %d p %d\n", mas_begin[0], mas_end[0], p);	
				} else {
					p = partition(a, mas_begin[1], mas_end[1]);

					mas_begin[i] = mas_begin[1];
					mas_end[i] = p - 1;	

					mas_begin[1] = p;
					mas_end[1] = N - 1;	
				}	
			
		}
		for (int i = 1; i < commsize; i++) { mas_begin[i], mas_end[i], mas_end[i] - mas_begin[i]);
			 MPI_Send(&a[mas_begin[i]], mas_end[i] - mas_begin[i] + 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);	
		}
		
		quick_sort(a, mas_begin[0], mas_end[0]);		
		for (int i = 1; i < commsize; i++) {
			MPI_Recv(&a[mas_begin[i]], mas_end[i] - mas_begin[i] + 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);	
		}	
		//printf("end ");
		//print_mas(a, 0, N - 1);
		
	}
	if (rank > 0 ) {
		double *b = malloc(sizeof(*b) * N);
		int buf_count;
		MPI_Recv(&b[0], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &Status);	
	
		MPI_Get_count(&Status, MPI_DOUBLE, &buf_count);
		quick_sort(b, 0, buf_count - 1);
		MPI_Send(&b[0], buf_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	
}

void check(double *a)
{
	for (int i = 0; i < N - 1; i++) {
		if (a[i] > a[i + 1]) printf("Verification is not succsessful!\n");
	}
}
int main(int argc, char **argv)
{
	int rank;
	double *a = malloc(sizeof(*a) * N);
	for (int i = 0; i < N; i++) {
		a[i] = getrand();
		//printf(" %.2f ", a[i]);
	}
	printf("\n");
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double t = MPI_Wtime();
	quick_sort_parallel(a, 0, N - 1);
	if(rank == 0) {
		check(a);
		t = MPI_Wtime() - t;
		printf("Time work: %.6f\n", t);
	}
	free(a);
	MPI_Finalize();
 	return 0;
}
