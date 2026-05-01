/*****************************************************************************/
// gcc -pthread -O1 test_conj_grad.c dot_product.c -lrt -lm -o test_conj_grad

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>

#include "dot_product.h"
#include "mat_vec_mult.h"
#include "matrix.h"
#include "vec_copy.h"
#include "vec_mul_add.h"
#include "vector.h"




#define CPNS 2.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

/* We want to test a wide range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
// #define A   2  /* coefficient of x^2 */
// #define B   40  /* coefficient of x */
// #define C   916  /* constant term */

#define NUM_TESTS 6   /* Number of different sizes to test */
#define OPTIONS 4




/* Prototypes */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
void check_answers(matrix_ptr a0, vector_ptr p0, vector_ptr b0, int N);
void conjugate_gradient_serial(int n, matrix_ptr A, vector_ptr b, vector_ptr x);
void conjugate_gradient_pthread(int n, matrix_ptr Ad, vector_ptr bd, vector_ptr x);
void conjugate_gradient_avx_vector(int n, matrix_ptr Ad, vector_ptr bd, vector_ptr xd);
void init_rand(matrix_ptr A, vector_ptr b, long int row_len);

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

/*************************************************/
/*
 * Inputs:
 *   n - dimension of the system (A is n x n, b and x are length n)
 *   A - matrix A
 *   b - right-hand-side vector b
 *   x - output: the solution vector x (written in place)
 *
 * Termination: loop exits when the residual L2 norm drops below 1e-10
 * or after max_it = 100 iterations, whichever comes first.
 *
 * Implementation notes (no optimizations applied):
 *   - All work runs on a single thread; no parallelism.
 *   - Matrix-vector product and dot products are delegated to serial
 *     kernels (mat_vec_mul_serial, dot_serial).
*/
void conjugate_gradient_serial(int n, matrix_ptr A, vector_ptr b, vector_ptr x)
{

  data_t rsold, rsnew, alpha, beta;
  
  int it = 0;
  int converged = 0;
  int max_it = 100;
  float tolerance = 1e-10;

  vector_ptr r = new_vector(n);
  zero_vector(r, n);

  vector_ptr p = new_vector(n);
  zero_vector(p, n);

  vector_ptr Ap = new_vector(n);
  zero_vector(Ap, n);

  data_t *x_ptr  = get_vector_start(x);
  data_t *r_ptr  = get_vector_start(r);
  data_t *p_ptr  = get_vector_start(p);
  data_t *Ap_ptr = get_vector_start(Ap);
  data_t *b_ptr  = get_vector_start(b);

  for (int i = 0; i < n; i++) {
    // Need to use following if x != 0:
    // double Ax0 = 0;
    // for (int j = 0; j < n; j++) Ax0 += A[i][j] * x[j];
    // r[i] = b[i] - Ax0;
    r_ptr[i] = b_ptr[i];
    p_ptr[i] = r_ptr[i]; // Initial search direction is the residual
  }
  
  rsold = dot_serial(n, r, r);

  while  (it < max_it)
  {

    // Compute Ap = A * p
    mat_vec_mul_serial(n, A, p, Ap);

    // alpha = rsold / (p' * A * p)
    alpha = rsold / dot_serial(n, p, Ap);

    // Update x and r
    for (int j = 0; j < n; j++) {
        x_ptr[j] = x_ptr[j] + alpha * p_ptr[j];
        r_ptr[j] = r_ptr[j] - alpha * Ap_ptr[j];
    }

    rsnew = dot_serial(n, r, r);

    if (sqrt(rsnew) < tolerance)
    {
      converged = 1;
      break;
    }

    // beta = rsnew / rsold
    beta = rsnew / rsold;
    // float rr = dot(r0,r0, N);
    for (int j = 0; j < n; j++) p_ptr[j] = r_ptr[j] + beta * p_ptr[j];

    rsold = rsnew;
     it++;

  }

  
  // if (converged) {
  //   printf("Converged after %d iterations\n", it);
  // } else {
  //     printf("Did not converge within %d iterations\n", max_it);
  // }

  // printf("CG solution: [%.10f, %.10f, %.10f, %.10f]\n", p[0], p[1], p[2], p[3]);
  // printf("CG Solution: \n");
//  print_vector(x);
}


/*************************************************/
/*
 * Inputs:
 *   n  - dimension of the system (A is n x n, b and x are length n)
 *   Ad - matrix A
 *   bd - right-hand-side vector b
 *   xd - output: the solution vector x (written in place)
 *
 * Termination: loop exits when the residual L2 norm drops below 1e-10
 * or after max_it = 100 iterations, whichever comes first.
 *
 * Parallelization:
 *   Each of the vector/matrix kernels (dot product, matvec, AXPY, copy)
 *   spawns NUM_THREADS pthreads internally, partitions the work across
 *   disjoint index ranges, and joins before returning. The CG loop itself
 *   runs serially on the main thread; parallelism is inside each kernel.
*/
void conjugate_gradient_pthread(int n, matrix_ptr Ad, vector_ptr bd, vector_ptr xd)
{
  int it = 0;
  int converged = 0;
  int max_it = 100;
  float tolerance = 1e-10;

  vector_ptr rd = new_vector(n);
  zero_vector(rd, n);

  vector_ptr pd = new_vector(n);
  zero_vector(pd, n);

  vector_ptr Apd = new_vector(n);
  zero_vector(Apd, n);

  data_t rsold, rsnew, pAp, alpha;

  // Initial r = b - Ax (assuming initial x is zeros)
  // r = b
  vec_copy_pthreads(n, bd, rd);

  // p = r (initial search direction)
  vec_copy_pthreads(n, rd, pd);

  // rsold = r · r
  rsold = dot_product_pthread_create(n, rd, rd);
 
  while  (it < max_it)
  { 
    // Compute Ap = A * p
    mat_vec_mul_pthreads_create(n, Ad, pd, Apd);

    // alpha = rsold / (p · Ap)
    pAp = dot_product_pthread_create(n, pd, Apd);

    alpha = rsold / pAp;

    // x = x + alpha * p
    vec_mul_add_pthreads_create(n, alpha, pd, xd, xd);  
   

    // r = r - alpha * Ap
    vec_mul_add_pthreads_create(n, -alpha, Apd, rd, rd);  


    // rsnew = r · r
    rsnew = dot_product_pthread_create(n, rd, rd);

    // Check convergence
    if (sqrt(rsnew) < tolerance)
    {
      converged = 1;
      break;
    }

    // beta = rsnew / rsold
    data_t beta = rsnew / rsold;

    // p = r + beta * p
    vec_mul_add_pthreads_create(n, beta, pd, rd, pd);  


    rsold = rsnew;

    it++;

  }

  if (converged) {
    printf("Converged after %d iterations\n", it);
  } else {
      printf("Did not converge within %d iterations\n", max_it);
  }

 // print_vector(xd);

}

void conjugate_gradient_openMP(int n, matrix_ptr Ad, vector_ptr bd, vector_ptr xd)
{
  int it = 0;
  int converged = 0;
  int max_it = 100;
  float tolerance = 1e-10;

  // print_matrix(Ad);
  // print_vector(bd);

  vector_ptr rd = new_vector(n);
  zero_vector(rd, n);

  vector_ptr pd = new_vector(n);
  zero_vector(pd, n);

  vector_ptr Apd = new_vector(n);
  zero_vector(Apd, n);

  data_t rsold, rsnew, pAp, alpha;

  // Initial r = b - Ax (assuming initial x is zeros)
  // r = b
  vec_copy_omp(n, bd, rd);

  // p = r (initial search direction)
  vec_copy_omp(n, rd, pd);

  // rsold = r · r
  rsold = dot_product_omp(n, rd, rd);


  while  (it < max_it)
  { 
    // Compute Ap = A * p
    mat_vec_mul_openmp(n, Ad, pd, Apd);

    // alpha = rsold / (p · Ap)
    pAp = dot_product_omp(n, pd, Apd);

    alpha = rsold / pAp;

    // x = x + alpha * p
    vec_mul_add_openmp(n, alpha, pd, xd, xd);  
   

    // r = r - alpha * Ap
    vec_mul_add_openmp(n, -alpha, Apd, rd, rd);  

    // print_vector(rd);

    // rsnew = r · r
    rsnew = dot_product_omp(n, rd, rd);

    // Check convergence
    if (sqrt(rsnew) < tolerance)
    {
      converged = 1;
      break;
    }

    // beta = rsnew / rsold
    data_t beta = rsnew / rsold;

    // p = r + beta * p
    vec_mul_add_openmp(n, beta, pd, rd, pd);  


    rsold = rsnew;

    it++;
  }

  // if (converged) {
  //   printf("Converged after %d iterations\n", it);
  // } else {
  //     printf("Did not converge within %d iterations\n", max_it);
  // }

//  print_vector(xd);
}


void conjugate_gradient_avx_vector(int n, matrix_ptr Ad, vector_ptr bd, vector_ptr xd)
{
  int it = 0;
  int converged = 0;
  int max_it = 100;
  float tolerance = 1e-10;

  vector_ptr rd = new_vector(n);
  zero_vector(rd, n);

  vector_ptr pd = new_vector(n);
  zero_vector(pd, n);

  vector_ptr Apd = new_vector(n);
  zero_vector(Apd, n);

  data_t rsold, rsnew, pAp, alpha;

  // Initial r = b - Ax (assuming initial x is zeros)
  // r = b
  vec_copy_avx_vectorize(n, bd, rd);
  
 
  // p = r (initial search direction)
  vec_copy_avx_vectorize(n, rd, pd);

  // rsold = r · r
  rsold = dot_product_avx_vectorize(n, rd, rd);
  // printf("rsold is %f\n", rsold);

 while  (it < max_it)
  { 
    // Compute Ap = A * p
    mat_vec_mul_avx_vectorize(n, Ad, pd, Apd);

    // alpha = rsold / (p · Ap)
    pAp = dot_product_avx_vectorize(n, pd, Apd);

    alpha = rsold / pAp;

    // x = x + alpha * p
    vec_mul_add_avx_vectorize(n, alpha, pd, xd, xd);  
   
    // print_vector(xd);

    // r = r - alpha * Ap
    vec_mul_add_avx_vectorize(n, -alpha, Apd, rd, rd);  

    // print_vector(rd);

    // rsnew = r · r
    rsnew = dot_product_avx_vectorize(n, rd, rd);

    // Check convergence
    if (sqrt(rsnew) < tolerance)
    {
      converged = 1;
      break;
    }

    // beta = rsnew / rsold
    data_t beta = rsnew / rsold;

    // p = r + beta * p
    vec_mul_add_avx_vectorize(n, beta, pd, rd, pd);  


    rsold = rsnew;

    it++;
  }

  // if (converged) {
  //   printf("Converged after %d iterations\n", it);
  // } else {
  //     printf("Did not converge within %d iterations\n", max_it);
  // }

//  print_vector(xd);

}
/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  //long int x, n;
  omp_set_num_threads(4);
  int z;
  z = NUM_TESTS-1;
  // long int alloc_size = 4;
  wakeup_answer = wakeup_delay();
  int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
  // int sizes[] = {16};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < num_sizes; s++)
  {
    int alloc_size = sizes[s];
    matrix_ptr A = new_matrix(alloc_size);
    vector_ptr b = new_vector(alloc_size);
    init_rand(A, b, alloc_size);

    vector_ptr x = new_vector(alloc_size);
    zero_vector(x, alloc_size);

    
    OPTION = 0;

    // conjugate_gradient_avx_vector(alloc_size, A, b, x);
   

    printf("Testing Conjugate Gradient Serial, size %d\n", alloc_size);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    conjugate_gradient_serial(alloc_size, A, b, x);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][s] = interval(time_start, time_stop);

    printf("\n");
  
    zero_vector(x, alloc_size);
    OPTION++;
  
      printf("Testing Conjugate Gradient Pthread, size %d\n", alloc_size);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      conjugate_gradient_pthread(alloc_size, A, b, x);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      time_stamp[OPTION][s] = interval(time_start, time_stop);
  

    printf("\n");
    zero_vector(x, alloc_size);

    OPTION++;
    if (OPTIONS > 2) {
    
        printf("Testing Conjugate Gradient OpenMP, size %d\n", alloc_size);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        conjugate_gradient_openMP(alloc_size, A, b, x);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        time_stamp[OPTION][s] = interval(time_start, time_stop);

    }

    printf("\n");
    zero_vector(x, alloc_size);
    OPTION++;
    if (OPTIONS > 2) {
      
        printf("Testing Conjugate Gradient AVX, size %d\n", alloc_size);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        conjugate_gradient_avx_vector(alloc_size, A, b, x);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        time_stamp[OPTION][s] = interval(time_start, time_stop);

    }

    free(A->data);
    free(A);
    free(b->data);
    free(b);
    free(x->data);
    free(x);
   
  }

  



  printf("Done collecting measurements.\n\n");

  printf("row_len, Serial, Pthread, OpenMP, AVX\n");
  for (int s = 0; s < num_sizes; s++) {
    printf("%d", sizes[s]);
    for (int j = 0; j < OPTIONS; j++) {
        printf(", %ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][s]));
    }
    printf("\n");
  }
  printf("\n");

  printf("Wakeup delay computed: %g \n", wakeup_answer);


} /* end main */

/**********************************************/




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


/*************************************************/




