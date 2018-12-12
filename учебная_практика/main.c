#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "tree.h"
int getrand(int min, int max)
{
    return (double)rand() / (RAND_MAX + 1.0) * (max - min) + min;
}
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}
int main(int argc, char *argv[]) {
	struct heap *h;
	double t;
	int i ,l;
	int max = 600000;
   struct heapnode node, *n;
	t = wtime();	
	h = heap_create(max);
	for (i = 0; i < max; i++) {
		l = getrand(0, max);
		heap_insert (h, l, l);
	}
	t = wtime()-t;
	printf("Time create heap %f %d\n",t, h->nnodes);
	t = wtime();
	node = heap_extract_min(h);	
	t = wtime()-t;
	printf("Delete min: %d time - %f\n", node.key, t);
	t = wtime();
	n = heap_min(h);	
	t = wtime()-t;
	l = n->key;
	printf("Search min: %d time - %f\n",l, t);
	t = wtime();
	i = heap_decrease_key(h, 200, 10);
	t = wtime() - t;
	printf("Time decrease_key %f\n", t);
	heap_free(h);
	return 0;
}
