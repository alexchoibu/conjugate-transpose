#ifndef VEC_MUL_ADD_H
#define VEC_MUL_ADD_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>

#include "vector.h"
#include "config.h"



typedef float data_t;

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define VBYTES 32
#define VSIZE (VBYTES/sizeof(data_t))

typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));

struct vm_thread_data{
  int thread_id;
  int n;
  data_t c;
  vector_ptr x;
  vector_ptr y;
  vector_ptr result;
};



void vec_mul_add_pthreads_create(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result);
void vec_mul_add_openmp(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result);
void vec_mul_add_avx_vectorize(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result);
#endif