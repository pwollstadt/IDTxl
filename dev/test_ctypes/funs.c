#include <stdio.h>
#include <math.h>
#include "funs.h"

// Berechnet die Fakultaet einer ganzen Zahl 
int fakultaet(int n) 
    { 
    int i; 
    int ret = 1; 
    for(i = 1; i <= n; i++) 
        ret *= i; 
    return ret; 
    }
    
// Berechnet die Laenge eines Vektors im R3 
double veclen(double x, double y, double z) 
    { 
    return sqrt(x*x + y*y + z*z); 
    }

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