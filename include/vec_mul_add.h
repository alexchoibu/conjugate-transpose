#ifndef VEC_MUL_ADD_H
#define VEC_MUL_ADD_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include "vector.h"


#define NUM_THREADS 2

typedef float data_t;



struct vm_thread_data{
  int thread_id;
  int n;
  data_t c;
  vector_ptr x;
  vector_ptr y;
  vector_ptr result;
};



void vec_mul_add_pthreads_create(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result);


#endif