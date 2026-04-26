#include "dot_product.h"



float dot_serial(int n, vector_ptr a, vector_ptr b) {
    float sum = 0.0;
    data_t *a_ptr = get_vector_start(a);
    data_t *b_ptr = get_vector_start(b);
    for (int i = 0; i < n; i++) {
        sum += a_ptr[i] * b_ptr[i];
    }
    return sum;
}

void *dot_product_pthread_func(void *threadarg) {
    
    struct d_thread_data *my_data; 
    my_data = (struct d_thread_data *) threadarg;
    int taskid = my_data->thread_id;
    int n = my_data->n;
  
    float sum = 0.0;
    data_t *a_ptr = get_vector_start(my_data->a);
    data_t *b_ptr = get_vector_start(my_data->b);

    int rows_per_thread = n / NUM_THREADS;
    int start_row, end_row;

    start_row = taskid * (rows_per_thread);
    end_row = start_row + rows_per_thread;

    for (int i = start_row; i < end_row; i++) {
        sum += a_ptr[i] * b_ptr[i];
    }

    my_data->sum = sum;

    pthread_exit(NULL);
}

float dot_product_pthread_create(int n, vector_ptr a, vector_ptr b)
{
  pthread_t threads[NUM_THREADS];
  struct d_thread_data thread_data_array[NUM_THREADS];
  int rc;

  for (long t = 0; t < NUM_THREADS; t++)
  {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].n = n;
    thread_data_array[t].a = a;
    thread_data_array[t].b = b;
    thread_data_array[t].sum = 0.0;
    rc = pthread_create(&threads[t], NULL, dot_product_pthread_func,
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

  float sum = 0.0;
  for (long t = 0; t < NUM_THREADS; t++)
  {
    sum += thread_data_array[t].sum;
  }

  return sum;

}
  
float dot_product_omp(int n, vector_ptr a, vector_ptr b) {
    float sum = 0.0;
    data_t *a_ptr = get_vector_start(a);
    data_t *b_ptr = get_vector_start(b);

#pragma omp parallel for reduction(+:sum)    
    for (int i = 0; i < n; i++) {
        sum += a_ptr[i] * b_ptr[i];
    }
    return (float )sum;
}


float dot_product_avx_vectorize(int n, vector_ptr a, vector_ptr b)
{
  long nLoop = n/4;
  __m128   m0 = _mm_setzero_ps();

  data_t *a_ptr = get_vector_start(a);
  data_t *b_ptr = get_vector_start(b);

  __m128*  pSrc1 = (__m128*) a_ptr;
  __m128*  pSrc2 = (__m128*) b_ptr;


  for (int i = 0; i < nLoop; i++) {
      m0 = _mm_add_ps(m0, _mm_dp_ps(*pSrc1, *pSrc2, 0xF1)); 

      pSrc1++;
      pSrc2++;
  }
  return _mm_cvtss_f32(m0);  

}