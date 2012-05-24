/*
 * si_omp.cpp
 *
 *  Created on: May 23, 2012
 *      Author: sanya-m
 */

#include <cstdio>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>
#include <sys/time.h>
#include <omp.h>

using namespace std;

enum ARGS {
	ARG_APPNAME, ARG_SIZE, ARG_T, ARG_E, ARG_C
};

class matrix {
	double *data;
	int _w, _h;
public:

	matrix(int h, int w) :
			_w(w), _h(h) {
		data = new double[_w * _h];
	}

	int w() const {
		return _w;
	}

	int h() const {
		return _h;
	}

	double *operator[](ptrdiff_t row) {
		return data + row * _w;
	}

	const double *operator[](ptrdiff_t row) const {
		return data + row * _w;
	}

	matrix &operator =(const matrix &r) {
		delete[] data;
		_w = r._w;
		_h = r._h;
		data = new double[_w * _h];
		copy(r.data, r.data + _w * _h, data);
		return *this;
	}

	matrix(const matrix &r) :
			_w(r._w), _h(r._h) {
		data = new double[_w * _h];
		copy(r.data, r.data + _w * _h, data);
	}

	~matrix() {
		delete[] data;
	}
};

int main(int argc, char *argv[]) {
	if (argc != ARG_C) {
		return EXIT_FAILURE;
	}

	int n = atoi(argv[ARG_SIZE]);
	if (n <= 0) {
		return EXIT_FAILURE;
	}

	double t = atof(argv[ARG_T]);
	if (t <= 0.0) {
		return EXIT_FAILURE;
	}

	double e = atof(argv[ARG_E]);
	if (e <= 0.0) {
		return EXIT_FAILURE;
	}

	matrix m(n, n + 1);
	double zf = 0.0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			m[i][j] = (i == j) ? 2.0 : 1.0;
		}
		m[i][n] = n + 1.0;
		zf += m[i][n] * m[i][n];
	}

	vector<double> x(n, 0.5), xold(n);
	double zsi;

	timeval tv1, tv2;
	gettimeofday(&tv1, NULL);

	e *= e;
#pragma omp parallel
	{
		do {
#pragma omp for schedule(static)
			for (int i = 0; i < n; i++) {
				xold[i] = x[i];
			}

			zsi = 0.0;
#pragma omp barrier

#pragma omp for reduction(+:zsi) schedule(guided)
			for (int i = 0; i < n; i++) {
				double si = 0.0;
				for (int j = 0; j < n; j++) {
					si += m[i][j] * xold[j];
				}
				si -= m[i][n];
				x[i] = xold[i] - t * si;
				zsi += si * si;
			}
#pragma omp barrier
		} while (fabs(zsi / zf) >= e);
	}

	gettimeofday(&tv2, NULL);
	int dt = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;

	for (int i = 0; i < n; i++) {
		printf("x[%d]=%lf\n", i, x[i]);
	}
	printf("Time: %lf\n", dt * 1e-6);
}
