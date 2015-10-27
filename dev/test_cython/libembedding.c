#include <stdio.h>

// time series embedding c 
void embed(double *embeddedPoints, double *ts_p, int ts_len, int dim, int tau, int delay)  
{ 
		
	int firstPredictionPoint = delay;
	int lastPredictionPoint = ts_len-dim*tau+1;
	int count_1 = 0;
	int count_2 = 0;
	int i, j;

	for (i = firstPredictionPoint; i < lastPredictionPoint+1; i++)
	{
		for (j=0; j<dim; j++)
		{			
			//printf("ind: %d (count_1=%d, count_2=%d\t)", count_1*dim + count_2, count_1, count_2);
			//printf("point: %.0f at %d into %d\n", ts_p[i + j*tau], i + j*tau,count_1*dim + count_2);
			embeddedPoints[count_1*dim + count_2] = ts_p[i + j*tau];
			count_2++;			
		}
		
		count_2 = 0;
		count_1++;			
	}

}

// test script for embedding code
/* 
int main(void)
{
	int ts_len = 10;
	int delay = 0;
	int dim = 3;
	int tau = 2;
	int nEmbeddedPoints = (ts_len - delay - dim*tau) + 2;
	double embeddedPoints[nEmbeddedPoints*dim];
	double ts[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9};

	int i,j;
	for (i=0; i<ts_len; i++)
	{
		printf("p %d: %.0f\t",i, ts[i]);
	}
	printf("\nnEmbedPoints %d:\n", nEmbeddedPoints);

	embed(embeddedPoints,ts, ts_len, dim, tau, delay);
	
	for (i=0; i<nEmbeddedPoints*dim; i++)
	{
	 		printf("%.0f\t", embeddedPoints[i]);
	}
}*/
