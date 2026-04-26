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

/* ================= MATRIX-VECTOR MULTIPLY ================= */
void matvec(int n, double *A, double *p, double *Ap)
{
    for (int row = 0; row < n; row++) {
        double sum = 0.0;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col] * p[col];
        }
        Ap[row] = sum;
    }
}

/* ================= VECTOR OPS ================= */

// x = x + alpha * p
void axpy_x(int n, double alpha, double *p, double *x)
{
    for (int i = 0; i < n; i++)
        x[i] += alpha * p[i];
}

// r = r - alpha * Ap
void axpy_r(int n, double alpha, double *Ap, double *r)
{
    for (int i = 0; i < n; i++)
        r[i] -= alpha * Ap[i];
}

// p = r + beta * p
void update_p(int n, double beta, double *r, double *p)
{
    for (int i = 0; i < n; i++)
        p[i] = r[i] + beta * p[i];
}

/* ================= CONJUGATE GRADIENT ================= */
void conjugate_gradient(int n, double *A, double *b, double *x, double tol)
{
    double *r  = (double*) malloc(n * sizeof(double));
    double *p  = (double*) malloc(n * sizeof(double));
    double *Ap = (double*) malloc(n * sizeof(double));

    double rsold, rsnew, alpha, beta;

    /* r = b - A*x (x = 0 initially in your test) */
    matvec(n, A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    rsold = dot_product(n, r, r);

    for (int iter = 0; iter < n; iter++) {

        /* Ap = A * p */
        matvec(n, A, p, Ap);

        /* alpha = rsold / (p^T Ap) */
        double pAp = dot_product(n, p, Ap);
        alpha = rsold / pAp;

        /* x and r updates */
        axpy_x(n, alpha, p, x);
        axpy_r(n, alpha, Ap, r);

        /* convergence check */
        rsnew = dot_product(n, r, r);
        if (sqrt(rsnew) < tol) break;

        beta = rsnew / rsold;

        /* update search direction */
        update_p(n, beta, r, p);

        rsold = rsnew;
    }

    free(r);
    free(p);
    free(Ap);
}

/* ================= MAIN ================= */
int main()
{
    int n = 2;
    double A[4] = {4, 1, 1, 3};
    double b[2] = {1, 2};
    double x[2] = {0};
    double tol = 1e-6;

    conjugate_gradient(n, A, b, x, tol);

    printf("Solution: ");
    for (int i = 0; i < n; i++)
        printf("%f ", x[i]);
    printf("\n");

    return 0;
}