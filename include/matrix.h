#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

typedef float data_t;
#define IDENT 0

typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;



matrix_ptr new_matrix(long int row_len);
int set_matrix_row_length(matrix_ptr m, long int row_len);
long int get_matrix_row_length(matrix_ptr m);
int init_matrix(matrix_ptr m, long int row_len, float *data_ptr);
int zero_matrix(matrix_ptr m, long int row_len);
data_t *get_matrix_start(matrix_ptr m);
void print_matrix(matrix_ptr m);


#endif