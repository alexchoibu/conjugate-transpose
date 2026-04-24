#ifndef VECTOR_H
#define VECTOR_H


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>



typedef float data_t;

typedef struct {
  long int len;
  data_t *data;
} vector_rec, *vector_ptr;






vector_ptr new_vector(long int row_len);
data_t *get_vector_start(vector_ptr v);
int set_vector_row_length(vector_ptr v, long int row_len);
long int get_vector_row_length(vector_ptr v);
int init_vector(vector_ptr v, long int row_len, float *data_ptr);
int zero_vector(vector_ptr v, long int row_len);
void print_vector(vector_ptr v);
#endif