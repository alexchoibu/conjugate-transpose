#ifndef VEC_COPY_H
#define VEC_COPY_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>

#include "vector.h"
#include "config.h"

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

typedef float data_t;

#define VBYTES 32
#define VSIZE (VBYTES/sizeof(data_t))

typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));


struct copy_arg {
    int     thread_id;
    int     n;
    vector_ptr x;
    vector_ptr y;
};

void vec_copy_serial(int n, vector_ptr x, vector_ptr y);
void vec_copy_pthreads(int n, vector_ptr x, vector_ptr y);
void vec_copy_omp(int n, vector_ptr x, vector_ptr y);
void vec_copy_avx_vectorize(int n, vector_ptr x, vector_ptr y);
#endif
