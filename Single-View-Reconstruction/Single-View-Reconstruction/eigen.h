#ifndef EIGEN_H
#define EIGEN_H

void eig_sys (int dimension, float **m, float **eig_vec, float *eig_val);
void tqli(int n, float *d,float *e, float **z);
void tred2(int n, float **a, float *d, float *e);

#endif