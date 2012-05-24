/*
 * mv1.cpp
 *
 *  Created on: May 23, 2012
 *      Author: sanya-m
 */

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include <mpi.h>

using namespace std;
using namespace MPI;

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

void MV1(Cartcomm comm, const vector<int> &rowcounts,
		const vector<int> &startrows, const matrix &m, vector<double> &v,
		vector<double> &vout) {

	int prev, next;
	comm.Shift(0, -1, prev, next);
	int rank;
	comm.Get_coords(comm.Get_rank(), 1, &rank);
	int size = comm.Get_size();

	int i = rank;
	for (;;) {
		for (int j = 0; j < m.h(); j++) {
			for (int k = 0; k < rowcounts[i]; k++) {
				vout[j] += m[j][startrows[i] + k] * v[k];
			}
		}
		i = (i + 1) % size;
		if (i == rank) {
			break;
		}
		comm.Sendrecv_replace(&v[0], v.size(), DOUBLE, next, 0, prev, 0);
	}
}

enum ARGS {
	ARG_APPNAME, ARG_SIZE, ARG_C
};

int main(int argc, char *argv[]) {
	Init(argc, argv);

	if (argc != ARG_C) {
		COMM_WORLD.Abort(EXIT_FAILURE);
	}
	int n = atoi(argv[ARG_SIZE]);
	if (n <= 0) {
		COMM_WORLD.Abort(EXIT_FAILURE);
	}

	int size = COMM_WORLD.Get_size();
	bool cyclic = true;
	Cartcomm comm = COMM_WORLD.Create_cart(1, &size, &cyclic, true);

	vector<int> rowcounts(size);
	int nrows = n;
	int maxrows = 0;
	for (int i = 0; i < size; i++) {
		rowcounts[i] = nrows / (size - i);
		maxrows = max(rowcounts[i], maxrows);
		nrows -= rowcounts[i];
	}
	vector<int> startrows(size);
	startrows[0] = 0;
	for (int i = 1; i < size; i++) {
		startrows[i] = startrows[i - 1] + rowcounts[i - 1];
	}

	int rank;
	comm.Get_coords(comm.Get_rank(), 1, &rank);
	matrix m(rowcounts[rank], n);
	fill(m[0], m[m.h()], 0.0);
	for (int i = 0; i < m.h(); i++) {
		m[i][n - i - startrows[rank] - 1] = 1.0;
	}

	vector<double> v(maxrows), vout(m.h(), 0.0);
	for (int i = 0; i < m.h(); i++) {
		v[i] = startrows[rank] + i;
	}

	comm.Barrier();
	double start = Wtime();

	MV1(comm, rowcounts, startrows, m, v, vout);

	comm.Barrier();
	double end = Wtime();

	vector<double> res(n);
	comm.Gatherv(&vout[0], vout.size(), DOUBLE, &res[0], &rowcounts[0], &startrows[0], DOUBLE, 0);
	if (rank == 0) {
		for (int i = 0; i < n; i++) {
			printf("v[%d]=%lf\n", i, res[i]);
		}
		printf("Time: %lf\n", end - start);
	}

	Finalize();
}
