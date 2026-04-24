#include "vec_copy.h"

void vec_copy_serial(int n, vector_ptr x, vector_ptr y)
{
  data_t *x_ptr = get_vector_start(x);
  data_t *y_ptr = get_vector_start(y);

  for (int i = 0; i < n; i++) {
    y_ptr[i] = x_ptr[i];
  }

}

void *vec_copy_pthread_func(void *threadarg)
{
  struct copy_arg *my_data; 
  my_data = (struct copy_arg *) threadarg;
  int taskid = my_data->thread_id;
  int n =  my_data->n;

  data_t *x_ptr = get_vector_start(my_data->x);
  data_t *y_ptr = get_vector_start(my_data->y);

  int rows_per_thread = n / NUM_THREADS;
  int start_row, end_row;

  start_row = taskid * (rows_per_thread);
  end_row = start_row + rows_per_thread;

  for (int i = start_row; i < end_row; i++) {
    y_ptr[i] = x_ptr[i];
  }
  pthread_exit(NULL); 
}

void vec_copy_pthreads(int n, vector_ptr x, vector_ptr y)
{
  pthread_t threads[NUM_THREADS];
  struct copy_arg thread_data_array[NUM_THREADS];
  int rc;

  for (long t = 0; t < NUM_THREADS; t++)
  {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].n = n;
    thread_data_array[t].x = x;
    thread_data_array[t].y = y;
    rc = pthread_create(&threads[t], NULL, vec_copy_pthread_func,
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