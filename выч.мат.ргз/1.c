#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;

double func (double x) {
	double t;
	//t = sin(x);
	//t = log(x);
	t = x*x;
	return t;
}
double InterpolateLagrangePolynomial (double x, double* x_values, double* y_values, int size)
{
	double lagrange_pol = 0;
	double basics_pol;

	for (int i = 0; i < size; i++)
	{
		basics_pol = 1;
		for (int j = 0; j < size; j++)
		{
			if (j == i) continue;
			basics_pol *= (x - x_values[j])/(x_values[i] - x_values[j]);		
		}
		lagrange_pol += basics_pol*y_values[i];
	}
	return lagrange_pol;
}
float inter1(float x0, float y0, float x1, float y1,float PX)
{
    float p = 0;
    p = (y0*(PX - x1)/(x0-x1)+y1*(PX - x0))/(x1-x0);
    return p;
}

float inter2(float x0, float y0, float x1, float y1, float x2, float y2, float PX)
{
    float p = 0;
    p = y0*((PX - x1)*(PX - x2))/((x0 - x1)*(x0 - x2))+y1*((PX - x0)*(PX - x2))/((x1-x0)*(x1 - x2)) + y2*((PX - x0)*(PX - x1))/((x2 - x0)*(x2-x1));
    return p;
}

int main() {
	int n = 0, i,st;
	double E,  L, x, a, b, L1;
	FILE *file;
	file = fopen("Equation.txt", "r");
	fscanf(file, "%lf", &x);
	fscanf(file, "%lf", &a);
	fscanf(file, "%lf", &b);
	fscanf(file, "%lf", &E);//шаг
	fscanf(file, "%d", &st);//степень
	L1 = x;
	if (b < a) {
		L1 = a;
		a = b;
		b = L1;
	}
	while (L1 < b) {
		L1 = L1 + E;
		n = n + 1;
	}
	double VectorX[n+1];
	double VectorY[n+1];
	for (i = 0; i <= n; i++) {
		VectorX[i] = a + i * E;
		VectorY[i] = func(VectorX[i]);
	}   
    printf("n\t");
    for (i=0; i<n; i++){
        printf("%d\t" , i);
    }
    printf ("\nx(n)\t");
	for (i = 0; i < n; i++) {
		printf("%.3lf\t", VectorX[i]);
	}
    printf ("\ny(n)\t");
	for (i = 0; i < n; i++) {
		printf("%.3lf\t",VectorY[i]);
	}
    printf ("\n\n");
	printf("x         \t L%d          \t f(x)      \n",st);
	for (x = a; x < b; x=x+0.5) {
		printf("%f \t",x);
		if(st == 1) { L = inter1(VectorX[0], VectorY[0],VectorX[1], VectorY[1], x); printf("%f\t",L);}
		if(st == 2) { L = inter2(VectorX[0], VectorY[0],VectorX[1], VectorY[1],VectorX[2], VectorY[2], x); printf("%f\t",L);}
		if(st > 2) {
			L = InterpolateLagrangePolynomial (x, VectorX, VectorY, st);
			printf("%f\t",L);	
		}
		printf("%f\n",func(x));
	}
	return 0;
}
