#include <stdio.h>
#include "funs.h"
#include <math.h>

int main(int argc, char* argv[]) {
	
  printf("\n\nTesting functions in C:\n");

  double v1, v2, v3;
  v1 = 3.5;
  v2 = 7.4;
  v3 = 1.2;
  
	
  int f1 = 4;
	
  double v_len = veclen(v1, v2, v3);	
	
  int f = 0;
  f = fakultaet(f1);

  printf("The faculty of %d is %d\n", f1, f);
  printf("The length of (%3.2f, %3.2f, %3.2f) is %3.2f\n", v1, v2, v3, v_len);
  

  int array_len = 5;
  int array[array_len];
  int i;
  for( i=0; i<array_len; i++ )	
	array[i] = i;
  sortiere(array, array_len);
  
  for( i=0; i<array_len; i++ )	
	printf("%d\n", array[i]);
	
  printf(" -- done \n\n");

  return 0;
}
