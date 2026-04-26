#ifndef MAT_VEC_MULT_H
#define MAT_VEC_MULT_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>

#include "vector.h"
#include "matrix.h"
#include "config.h"


typedef float data_t;

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define VBYTES 32
#define VSIZE (VBYTES/sizeof(data_t))

typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));


struct mv_thread_data{
  int thread_id;
  int n;
  matrix_ptr A;
  vector_ptr x;
  vector_ptr result;
};



void mat_vec_mul_serial(int n, matrix_ptr A, vector_ptr x, vector_ptr result);
void mat_vec_mul_pthreads_create(int n, matrix_ptr A, vector_ptr x, vector_ptr result);
void mat_vec_mul_openmp(int n, matrix_ptr A, vector_ptr x, vector_ptr result);
void mat_vec_mul_avx_vectorize(int n, matrix_ptr A, vector_ptr x, vector_ptr result);




#endif