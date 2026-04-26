#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ================= DOT PRODUCT ================= */
double dot_product(int n, double *a, double *b)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/* ================= MATVEC ================= */
void matvec(int n, double *A, double *p, double *Ap)
{
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * p[j];
        Ap[i] = sum;
    }
}

/* ================= AXPY OPS ================= */
void axpy_x(int n, double alpha, double *p, double *x)
{
    for (int i = 0; i < n; i++)
        x[i] += alpha * p[i];
}

void axpy_r(int n, double alpha, double *Ap, double *r)
{
    for (int i = 0; i < n; i++)
        r[i] -= alpha * Ap[i];
}

void update_p(int n, double beta, double *r, double *p)
{
    for (int i = 0; i < n; i++)
        p[i] = r[i] + beta * p[i];
}

/* ================= CG ================= */
void cg(int n, double *A, double *b, double *x)
{
    double *r  = (double*) malloc(n * sizeof(double));
    double *p  = (double*) malloc(n * sizeof(double));
    double *Ap = (double*) malloc(n * sizeof(double));

    double rsold, rsnew, alpha, beta;

    matvec(n, A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    rsold = dot_product(n, r, r);

    /* FIXED ITERATION COUNT for profiling consistency */
    int max_iters = 100;

    for (int iter = 0; iter < max_iters; iter++) {

        matvec(n, A, p, Ap);

        double pAp = dot_product(n, p, Ap);
        alpha = rsold / pAp;

        axpy_x(n, alpha, p, x);
        axpy_r(n, alpha, Ap, r);

        rsnew = dot_product(n, r, r);
        beta = rsnew / rsold;

        update_p(n, beta, r, p);

        rsold = rsnew;
    }

    free(r);
    free(p);
    free(Ap);
}

/* ================= MAIN (MATCH CUDA SIZES) ================= */
int main()
{
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = 6;

    for (int s = 0; s < num_sizes; s++) {

        int n = sizes[s];

        printf("\n===== Testing %d x %d =====\n", n, n);

        double *A = (double*) malloc(n * n * sizeof(double));
        double *b = (double*) malloc(n * sizeof(double));
        double *x = (double*) calloc(n, sizeof(double));

        /* Same SPD-style initialization as CUDA */
        for (int i = 0; i < n * n; i++)
            A[i] = ((double)rand() / RAND_MAX) * 10.0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double sym = 0.5 * (A[i*n + j] + A[j*n + i]);
                A[i*n + j] = sym;
                A[j*n + i] = sym;
            }
        }

        for (int i = 0; i < n; i++) {
            A[i*n + i] += n;
            b[i] = ((double)rand() / RAND_MAX) * 10.0;
        }

        cg(n, A, b, x);

        printf("done size %d\n", n);

        free(A);
        free(b);
        free(x);
    }

    return 0;
}