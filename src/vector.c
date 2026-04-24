#include "vector.h"

vector_ptr new_vector(long int row_len)
{
  long int i;
  long int alloc;

  /* Allocate and declare header structure */
  vector_ptr result = (vector_ptr) malloc(sizeof(vector_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = row_len;

  /* Allocate and declare array */
  if (row_len > 0) {
    result->data = (data_t *) calloc(row_len, sizeof(data_t));
    if (!result->data) { free(result); return NULL; }
  } else {
    result->data = NULL;
  }
  return result;

}
data_t *get_vector_start(vector_ptr v)
{
  return v->data;
}

int set_vector_row_length(vector_ptr v, long int row_len)
{
  v->len = row_len;
  return 1;
}
long int get_vector_row_length(vector_ptr v)
{
   return v->len;
}
int init_vector(vector_ptr v, long int row_len, float *data_ptr)
{
  if (row_len <= 0) return 0;
  v->len = row_len;
  for (long int i = 0; i < row_len; i++) {
    v->data[i] = *data_ptr;
    data_ptr++;
  }
  return 1;
}

int zero_vector(vector_ptr v, long int row_len)
{
  if (row_len <= 0) return 0;
  v->len = row_len;
  for (long int i = 0; i < row_len; i++) {
    v->data[i] = 0;
  }
  return 1;
}
void print_vector(vector_ptr v) 
{
  long int n = v->len;
  for (long int i = 0; i < n; i++) {
    printf("%8.3f\n", v->data[i]);
  }
  printf("\n");
}