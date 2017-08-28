#include "math.h"

/* $Id: modwtj.c 300 2004-03-18 02:15:10Z ccornish $ */

void modwtj(double *Vin, int N, int j, int L, double *ht, double *gt,
	          double *Wout, double *Vout)
{

  int k, n, t;

  for (t = 0; t < N; t++) {
    k = t;
    Wout[t] = ht[0] * Vin[k];
    Vout[t] = gt[0] * Vin[k];
    for (n = 1; n < L; n++) {
      k -= (int) pow(2.0, (double) j - 1.0);
      if (k < 0) {
	        k += N;
      }
      Wout[t] += ht[n] * Vin[k];
      Vout[t] += gt[n] * Vin[k];
    }
  }

}

void imodwtj(double *Win, double *Vin, int *N, int *j, int *L,
             double *ht, double *gt, double *Vout)
{

  int k, n, t;

  for(t = 0; t < *N; t++) {
    k = t;
    Vout[t] = (ht[0] * Win[k]) + (gt[0] * Vin[k]);
    for(n = 1; n < *L; n++) {
      k += (int) pow(2.0, (double) *j - 1.0);
      if(k >= *N) k -= *N;
          Vout[t] += (ht[n] * Win[k]) + (gt[n] * Vin[k]);
    }
  }
}

void test_c(int *a, int a_len)
{
  int i;
  for(i = 0; i < a_len; i++) {
      a[i] = i;
  }
}