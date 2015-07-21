#include <stdio.h>

// Bubblesort 
void sortiere(int *array, int len) 
    { 
    int i, j, tmp; 
    for(i = 0; i < len; i++) 
        { 
        for(j = 0; j < i; j++) 
            { 
            if(array[j] > array[i]) 
                { 
                tmp = array[j]; 
                array[j] = array[i]; 
                array[i] = tmp; 
                } 
            } 
        } 
    }

int main(int argc, char* argv[]) {

  int array_len = 5;
  int array[array_len];
  int i;
  for( i=0; i<array_len; i++ )	
	array[i] = i;
  sortiere(array, array_len);

  return 0;
}
