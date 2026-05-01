#include "mat_vec_mult.h"

/* mmm serial*/
void mat_vec_mul_serial(int n, matrix_ptr A, vector_ptr x, vector_ptr result)
{
  int i, j;

  data_t *A_ptr = get_matrix_start(A);
  data_t *x_ptr = get_vector_start(x);
  data_t *result_ptr = get_vector_start(result);
  
  for (i = 0; i < n; i++) {
    data_t sum = 0.0;
    for (j = 0; j < n; j++) {
      sum += A_ptr[i * n + j] * x_ptr[j];
    }
    result_ptr[i] = sum;
  }
}

void *mat_vec_mul_pthread_func(void *threadarg)
{
    struct mv_thread_data *my_data; 
    my_data = (struct mv_thread_data *) threadarg;
    int taskid = my_data->thread_id;
    int n =  my_data->n;

    data_t *A_ptr = get_matrix_start(my_data->A);
    data_t *x_ptr = get_vector_start(my_data->x);
    data_t *result_ptr = get_vector_start(my_data->result);

    int rows_per_thread = n / NUM_THREADS;
    int start_row, end_row;

    start_row = taskid * (rows_per_thread);
    end_row = start_row + rows_per_thread;

    for (int i = start_row; i < end_row; i++) {
        data_t sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A_ptr[i * n + j] * x_ptr[j];
        }
        result_ptr[i] = sum;
    }
    pthread_exit(NULL); 
}

void mat_vec_mul_pthreads_create(int n, matrix_ptr A, vector_ptr x, vector_ptr result)
{
  pthread_t threads[NUM_THREADS];
  struct mv_thread_data thread_data_array[NUM_THREADS];
  int rc;

  for (long t = 0; t < NUM_THREADS; t++)
  {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].n = n;
    thread_data_array[t].A = A;
    thread_data_array[t].x = x;
    thread_data_array[t].result = result;
    rc = pthread_create(&threads[t], NULL, mat_vec_mul_pthread_func,
                                                (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  for (long t = 0; t < NUM_THREADS; t++) {
    if (pthread_join(threads[t], NULL)) {
      exit(19);
    }
  }
}

void mat_vec_mul_openmp(int n, matrix_ptr A, vector_ptr x, vector_ptr result)
{
  int i, j;

  data_t *A_ptr = get_matrix_start(A);
  data_t *x_ptr = get_vector_start(x);
  data_t *result_ptr = get_vector_start(result);

#pragma omp parallel for
  for (i = 0; i < n; i++) {
    data_t sum = 0.0;
    for (j = 0; j < n; j++) {
      sum += A_ptr[i * n + j] * x_ptr[j];
    }
    result_ptr[i] = (data_t) sum;
  }
}

void mat_vec_mul_avx_vectorize(int n, matrix_ptr A, vector_ptr x, vector_ptr result)
{
  long nLoop = n/8;

  int i, j;

  data_t *A_ptr = get_matrix_start(A);
  data_t *x_ptr = get_vector_start(x);
  data_t *result_ptr = get_vector_start(result);

  for (int i = 0; i < n; i++) {

      data_t *row_ptr = &A_ptr[i * n];
      __m256 m0 = _mm256_setzero_ps();

      for (int j = 0; j < nLoop; j++) {
          __m256 va = _mm256_loadu_ps(row_ptr + j*8);
          __m256 vb = _mm256_loadu_ps(x_ptr   + j*8);
          __m256 prod = _mm256_mul_ps(va, vb);
          m0 = _mm256_add_ps(m0, prod);
      }

      float buf[8];
      _mm256_storeu_ps(buf, m0);
      result_ptr[i] = buf[0] + buf[1] + buf[2] + buf[3]
                    + buf[4] + buf[5] + buf[6] + buf[7];

  }
}