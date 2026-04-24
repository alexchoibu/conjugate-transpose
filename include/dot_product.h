#ifndef DOT_PROD_H
#define DOT_PROD_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include "vector.h"

#define NUM_THREADS 2

typedef float data_t;

struct d_thread_data{
  int thread_id;
  int n;
  vector_ptr a;
  vector_ptr b;
  data_t sum;
};


float dot_serial(int n, vector_ptr a, vector_ptr b);
float dot_product_pthread_create(int n, vector_ptr a, vector_ptr b);

#endif