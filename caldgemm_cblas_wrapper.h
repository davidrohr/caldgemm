/**
 * This file is part of the CALDGEMM library.
 *
 * Copyright 2015:
 *  - David Rohr (drohr@jwdt.org)
 *  - Matthias Bach (bach@compeng.uni-frankfurt.de)
 *  - Matthias Kretz (kretz@compeng.uni-frankfurt.de)
 *
 * This file is part of CALDGEMM.
 *
 * CALDGEMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CALDGEMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CALDGEMM.  If not, see <http://www.gnu.org/licenses/>.
 */

void cblas_dtrsma(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side, enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag, blasint M, blasint N, double alpha, double *A, blasint lda, double *B, blasint ldb);
void cblas_dgemva(enum CBLAS_ORDER order,  enum CBLAS_TRANSPOSE trans,  blasint m, blasint n, double alpha, double  *a, blasint lda,  double  *x, blasint incx,  double beta,  double  *y, blasint incy);
void cblas_dgemma(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, blasint M, blasint N, blasint K, double alpha, double *A, blasint lda, double *B, blasint ldb, double beta, double *C, blasint ldc);
void cblas_daxpya(blasint n, double, double *x, blasint incx, double *y, blasint incy);
void cblas_dscala(blasint N, double alpha, double *X, blasint incX);
