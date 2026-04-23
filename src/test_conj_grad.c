/*****************************************************************************/
// gcc -pthread -O1 test_conj_grad.c -lrt -lm -o test_conj_grad

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#define CPNS 2.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

/* We want to test a wide range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
// #define A   2  /* coefficient of x^2 */
// #define B   40  /* coefficient of x */
// #define C   916  /* constant term */

#define NUM_TESTS 1   /* Number of different sizes to test */
#define NUM_THREADS 8
#define OPTIONS 1
#define IDENT 0
/*
long int alloc_size = 2;
float a0_data[] = {4, 1, 1, 3};
float b0_data[] = {1,2};
int row_a = 2;
int col_a = 2;
*/
typedef float data_t;


long int alloc_size = 512;




/* Create abstract data type for matrix */
typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;

typedef struct {
  long int len;
  data_t *data;
} vector_rec, *vector_ptr;

struct thread_data{
  int thread_id;
  int n;
  matrix_ptr A;
  vector_ptr x;
  vector_ptr a;
  vector_ptr b;
  vector_ptr result;
  float sum;
};

struct thread_data_dot{
  int thread_id;
  vector_ptr a;
  vector_ptr b;
  int start;
  int end;
  int num_elements_per_thread;
  float sum;
};


/* Prototypes */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
data_t *get_vector_start(vector_ptr v);
matrix_ptr new_matrix(long int row_len);
int set_matrix_row_length(matrix_ptr m, long int row_len);
long int get_matrix_row_length(matrix_ptr m);
int init_matrix(matrix_ptr m, long int row_len, float *data_ptr);
int zero_matrix(matrix_ptr m, long int row_len);
void init_rand(matrix_ptr A, vector_ptr b, long int row_len);

vector_ptr new_vector(long int row_len);
int set_vector_row_length(vector_ptr v, long int row_len);
long int get_vector_row_length(vector_ptr v);
int init_vector(vector_ptr v, long int row_len, float *data_ptr);
int zero_vector(vector_ptr v, long int row_len);
void print_vector(vector_ptr v);
data_t *get_matrix_start(matrix_ptr m);
void mat_vec_mul_serial(int n, matrix_ptr A, vector_ptr x, vector_ptr result);
void conjugate_gradient(vector_ptr r0, matrix_ptr a0, vector_ptr d0, vector_ptr Ad, vector_ptr pnew0, vector_ptr p0, int N);
// void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
// void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void print_matrix(matrix_ptr m);
/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

void check_answers(matrix_ptr a0, vector_ptr p0, vector_ptr b0, int N)
{
  data_t *p = get_vector_start(p0);
  data_t *A = get_matrix_start(a0);
  data_t *b = get_vector_start(b0);

  double err2 = 0.0;
  for (int i = 0; i < N; i++) {
      double Ap_i = 0.0;
      for (int j = 0; j < N; j++) {
          Ap_i += A[i * N + j] * p[j];   // row i of A times p
      }
      double diff = Ap_i - b[i];         // should be ~0 if correct
      err2 += diff * diff;
  }
  printf("||A*p - b|| = %e\n", sqrt(err2));

  if (sqrt(err2) < 1e-5)
  {
    printf("PASSED\n");
  }
  else
  {
    printf("DID NOT PASS\n");
  }
}

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  long int x, n;


  x = NUM_TESTS-1;
  


  wakeup_answer = wakeup_delay();

  
  // float *a0_data_ptr = a0_data;
  /* declare and initialize the matrix structure */
  matrix_ptr a0 = new_matrix(alloc_size);
  // init_matrix(a0, alloc_size, a0_data_ptr);

  
  
  // float *b0_data_ptr = b0_data;
  vector_ptr b0 = new_vector(alloc_size);
  // init_vector(b0, alloc_size, b0_data_ptr);

  init_rand(a0, b0, alloc_size);

  // print_matrix(a0);
  // printf("\n");
  // print_vector(b0);

  
  // exit(1);
  vector_ptr p0 = new_vector(alloc_size);
  zero_vector(p0, alloc_size);
  // printf("\n");
  // print_vector(p0);
  
  
  vector_ptr Ad = new_vector(alloc_size);
  zero_vector(Ad, alloc_size);

  vector_ptr r0 = new_vector(alloc_size);
  zero_vector(r0, alloc_size);

  vector_ptr d0 = new_vector(alloc_size);
  zero_vector(d0, alloc_size);
  
  vector_ptr pnew = new_vector(alloc_size);
  zero_vector(pnew, alloc_size);

  mat_vec_mul_serial(alloc_size, a0, p0, Ad);
  // print_vector(Ad);
  
  data_t *r1 = get_vector_start(r0);
  data_t *b1 = get_vector_start(b0);
  data_t *Ad1 = get_vector_start(Ad);
  data_t *d1 = get_vector_start(d0);

  for (int i = 0; i < alloc_size; i++) {
        r1[i] = b1[i] - Ad1[i];
        d1[i] = r1[i];
  }

  // print_vector(r0);
 
  
//   matrix_ptr c0 = new_matrix(alloc_size);
//   zero_matrix(c0, alloc_size);

  OPTION = 0;
  x = 0;
//   for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    // printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    // set_matrix_row_length(a0, n);
    // set_matrix_row_length(b0, n);
    // set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    conjugate_gradient(r0, a0, d0, Ad, pnew, p0, alloc_size);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
//   }

  // After CG finishes, p holds the solution
 // After CG finishes, p holds the solution
  check_answers(a0, p0, b0, alloc_size);
  printf("Done collecting measurements.\n\n");

  printf("row_len, conj origonal\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
   //   printf("%ld, ", A*i*i + B*i + C);
      printf("%ld, ", alloc_size);
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int) ((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }
  printf("\n");

//   printf("Wakeup delay computed: %g \n", wakeup_answer);
} /* end main */

/**********************************************/

/* Create matrix of specified length */
matrix_ptr new_matrix(long int row_len)
{
  long int i;
  long int alloc;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = row_len;

  /* Allocate and declare array */
  if (row_len > 0) {
    alloc = row_len * row_len;
    data_t *data = (data_t *) calloc(alloc, sizeof(data_t));
    if (!data) {
	  free((void *) result);
	  printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                                       alloc * sizeof(data_t));
	  return NULL;  /* Couldn't allocate storage */
	}
	result->data = data;
  } else {
    result->data = NULL;
  }

  return result;
}

/* Set length of matrix */
int set_matrix_row_length(matrix_ptr m, long int row_len)
{
  m->len = row_len;
  return 1;
}

/* Return length of matrix */
long int get_matrix_row_length(matrix_ptr m)
{
  return m->len;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int row_len, float *data_ptr)
{
  long int i;

  if (row_len > 0) {
    m->len = row_len;
    for (i = 0; i < row_len*row_len; i++) {
    //  m->data[i] = (data_t)(i);
      m->data[i] = *data_ptr;
      data_ptr++;
    }
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int row_len)
{
  long int i,j;

  if (row_len > 0) {
    m->len = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      m->data[i] = (data_t)(IDENT);
    }
    return 1;
  }
  else return 0;
}

void init_rand(matrix_ptr A, vector_ptr b, long int row_len)
{
  srand(42);
  int N = row_len;
  A->len = row_len;
  
  for (int i = 0; i < N; i++) {
      for (int j = i + 1; j < N; j++) {
          data_t val = (data_t)rand() / RAND_MAX;  // in [0, 1]
          A->data[i * N + j] = val;
          A->data[j * N + i] = val;  // symmetry
      }
  }

  for (int i = 0; i < N; i++) {
    A->data[i * N + i] = (data_t)N;
  }
  b->len = row_len; 
  for (int i = 0; i < N; i++) {
    b->data[i] = (data_t)rand() / RAND_MAX;
  }

}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

void print_matrix(matrix_ptr m) {
  long int n = m->len;
  data_t *data = m->data;
  for (long int i = 0; i < n; i++) {
    for (long int j = 0; j < n; j++) {
      printf("%8.3f ", data[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}
/*************************************************/
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
void print_vector(vector_ptr v) {
  long int n = v->len;
  for (long int i = 0; i < n; i++) {
    printf("%8.3f\n", v->data[i]);
  }
  printf("\n");
}

/*************************************************/

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
    struct thread_data *my_data; 
    my_data = (struct thread_data *) threadarg;
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
  struct thread_data thread_data_array[NUM_THREADS];
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

/*************************************************/
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
    
    struct thread_data *my_data; 
    my_data = (struct thread_data *) threadarg;
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
  struct thread_data thread_data_array[NUM_THREADS];
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

/*************************************************/

void conjugate_gradient(vector_ptr r0, matrix_ptr a0, vector_ptr d0, vector_ptr Ad, vector_ptr pnew0, vector_ptr p0, int N)
{

  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  int it = 0;
  int converged = 0;
  int max_it = 100;
  float tolerance = 1e-10;
  data_t *r = get_vector_start(r0);
  data_t *pnew = get_vector_start(pnew0);
  data_t *d = get_vector_start(d0);
  data_t *p = get_vector_start(p0);
  data_t *Ad_p  = get_vector_start(Ad);
  printf("in conjugate grad function\n");
  
  while  (it < max_it)
  {
    // float rr = dot(r0,r0, N);
    float rr = dot_product_pthread_create(N, r0, r0);
    // exit(1);
    // printf("rr is %f\n", rr);
    // exit(1);
    if (sqrt(rr) < tolerance)
    {
      converged = 1;
      break;
    }
    // if (it == 1) exit(1);
    /*
    for (long t = 0; t < NUM_THREADS; t++)
    {
      thread_data_array[t].thread_id = t;

      rc = pthread_create(&threads[t], NULL, mat_vec_mul_pthreads,
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
       */
    // mat_vec_mul_serial(N, a0, d0, Ad);
    mat_vec_mul_pthreads_create(N, a0, d0, Ad);

  
    // print_vector(Ad);
    float alpha = rr / dot_product_pthread_create(N, d0, Ad);
    // print_matrix(a0);
    // print_vector(d0);
  
    // printf("alpha = %f %f\n", alpha, dot(d0, Ad, N));
  
   
    for (int i = 0; i < N; i++) {
      pnew[i] = p[i] + alpha * d[i];
      r[i]    = r[i] - alpha * Ad_p[i];
    }
    // print_vector(pnew0);
    // print_vector(r0);
   
    float beta = dot_product_pthread_create(N, r0, r0) / rr;
    // printf("beta is %f\n", beta);
    
    for (int i = 0; i < N; i++) {
        d[i] = r[i] + beta * d[i];
    }
    // print_vector(d0);
  
    for (int i = 0; i < N; i++) {
      p[i] = pnew[i];
    }
    // print_vector(p0);

    it++;
  }

  if (converged) {
    printf("Converged after %d iterations\n", it);
  } else {
      printf("Did not converge within %d iterations\n", max_it);
  }

  // printf("CG solution: [%.10f, %.10f, %.10f, %.10f]\n", p[0], p[1], p[2], p[3]);
  // printf("CG Solution: \n");
  // print_vector(p0);
}