#include "vec_mul_add.h"

void *vec_mul_add_pthread_func(void *threadarg)
{
    struct vm_thread_data *my_data; 
    my_data = (struct vm_thread_data *) threadarg;
    int taskid = my_data->thread_id;
    int n =  my_data->n;

    data_t c = my_data->c;
    data_t *x_ptr = get_vector_start(my_data->x);
    data_t *y_ptr = get_vector_start(my_data->y);
    data_t *result_ptr = get_vector_start(my_data->result);

    int rows_per_thread = n / NUM_THREADS;
    int start_row, end_row;

    start_row = taskid * (rows_per_thread);
    end_row = start_row + rows_per_thread;

    //result = c*x + y
    for (int i = start_row; i < end_row; i++) {
        
        result_ptr[i] = c * x_ptr[i] + y_ptr[i];
    }
    pthread_exit(NULL); 
}


void vec_mul_add_pthreads_create(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result)
{
  pthread_t threads[NUM_THREADS];
  struct vm_thread_data thread_data_array[NUM_THREADS];
  int rc;

  for (long t = 0; t < NUM_THREADS; t++)
  {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].n = n;
    thread_data_array[t].c = c;
    thread_data_array[t].x = x;
    thread_data_array[t].y = y;
    thread_data_array[t].result = result;
    rc = pthread_create(&threads[t], NULL, vec_mul_add_pthread_func,
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


void vec_mul_add_openmp(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result)
{

    data_t *x_ptr = get_vector_start(x);
    data_t *y_ptr = get_vector_start(y);
    data_t *result_ptr = get_vector_start(result);


#pragma omp parallel for
    for (int i = 0; i < n; i++) {   
      result_ptr[i] = c * x_ptr[i] + y_ptr[i];
    }
   
}

void vec_mul_add_avx_vectorize(int n, data_t c, vector_ptr x, vector_ptr y, vector_ptr result)
{
  long nLoop = n/8;

  int i, j;

  data_t *x_ptr = get_vector_start(x);
  data_t *y_ptr = get_vector_start(y);
  data_t *result_ptr = get_vector_start(result);

   __m256 vc = _mm256_set1_ps(c);

   

    for (long i = 0; i < nLoop; i++) {
   
      __m256 vx   = _mm256_loadu_ps(x_ptr + i*8);
      __m256 vy   = _mm256_loadu_ps(y_ptr + i*8);
      __m256 prod = _mm256_mul_ps(vc, vx);
      __m256 res  = _mm256_add_ps(prod, vy);
      _mm256_storeu_ps(result_ptr + i*8, res);
    }

}