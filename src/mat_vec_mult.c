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
  long nLoop = n/4;

  int i, j;

  data_t *A_ptr = get_matrix_start(A);
  data_t *x_ptr = get_vector_start(x);
  data_t *result_ptr = get_vector_start(result);

  for (int i = 0; i < n; i++) {
      // Each row of A starts at A_ptr + i*n
      // x is the same vector every iteration
      __m128* pRow = (__m128*) &A_ptr[i * n];
      __m128* pVec = (__m128*) x_ptr;

      __m128 m0 = _mm_setzero_ps();

      for (int j = 0; j < nLoop; j++) {
          m0 = _mm_add_ps(m0, _mm_dp_ps(*pRow, *pVec, 0xF1));
          pRow++;
          pVec++;
      }

      result_ptr[i] = _mm_cvtss_f32(m0);
  }
}