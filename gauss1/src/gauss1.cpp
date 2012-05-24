#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <valarray>
#include <vector>
#include <utility>
#include <algorithm>

using namespace std;
using namespace MPI;

enum ARGS {
	ARG_APPNAME, ARG_SIZE, ARG_C
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

	matrix(const matrix &r) : _w(r._w), _h(r._h) {
		data = new double[_w * _h];
		copy(r.data, r.data + _w * _h, data);
	}

	~matrix() {
		delete[] data;
	}
};

int rank;
int n;
matrix m(1, 1);
vector<int> rowc, frow;

void load_partition() {
	int nrows = n;
	int nproc = COMM_WORLD.Get_size();
	rowc.resize(nproc);
	frow.resize(nproc);

	for (int i = 0; i < nproc; i++) {
		rowc[i] = nrows / (nproc - i);
		nrows -= rowc[i];
	}
	frow[0] = 0;
	for (int i = 1; i < nproc; i++) {
		frow[i] = frow[i - 1] + rowc[i - 1];
	}
}

void gen() {
	m = matrix(rowc[rank], n + 1);
	fill(m[0], m[m.h()], 1.0);
	for (int i = 0; i < m.h(); i++) {
		m[i][i + frow[rank]] = 2.0;
		m[i][n] = n + 1.0;
	}
}

int main(int argc, char *argv[]) {
	Init(argc, argv);
	rank = COMM_WORLD.Get_rank();

	if (argc != ARG_C) {
		COMM_WORLD.Abort(EXIT_FAILURE);
	}

	n = atoi(argv[ARG_SIZE]);
	if (n <= 0) {
		COMM_WORLD.Abort(EXIT_FAILURE);
	}

	load_partition();
	gen();

	COMM_WORLD.Barrier();
	double start = Wtime();
	printf("%d start\n", rank);

	valarray<double> row(0.0, n + 1);
	for (int i = 0; i < frow[rank]; i++) {
		COMM_WORLD.Recv(&row[i], n + 1 - i, DOUBLE, ANY_SOURCE, i);

		for (int j = 0; j < rowc[rank]; j++) {
			double a = m[j][i] / row[i];
			for (int k = i; k < n + 1; k++) {
				m[j][k] -= a * row[k];
			}
		}
	}
	for (int rid = 0; rid < rowc[rank]; rid++) {
		for (int rid2 = 0; rid2 < rid; rid2++) {
			int j = rid2 + frow[rank];
			double a = m[rid][j] / m[rid2][j];
			for (int k = j; k < n + 1; k++) {
				m[rid][k] -= a * m[rid2][k];
			}
		}
		int i = frow[rank] + rid;
		for (int dest = rank + 1; dest < COMM_WORLD.Get_size(); dest++) {
			COMM_WORLD.Isend(&m[rid][i], n + 1 - i, DOUBLE, dest, i);
		}
	}

	printf("%d back\n", rank);

	for (int i = n - 1; i >= frow[rank] + rowc[rank]; i--) {
		double f;
		COMM_WORLD.Recv(&f, 1, DOUBLE, ANY_SOURCE, i);
		for (int j = 0; j < rowc[rank]; j++) {
			m[j][n] -= f * m[j][i];
		}
	}
	for (int rid = rowc[rank] - 1; rid >= 0; rid--) {
		for (int rid2 = rowc[rank] - 1; rid2 > rid; rid2--) {
			int j = rid2 + frow[rank];
			m[rid][n] -= m[rid2][n] * m[rid][j];
		}
		int i = frow[rank] + rid;
		m[rid][n] /= m[rid][i];
		for (int dest = rank - 1; dest >= 0; dest--) {
			COMM_WORLD.Isend(&m[rid][n], 1, DOUBLE, dest, i);
		}
	}
	printf("%d done\n", rank);

	COMM_WORLD.Barrier();
	double end = Wtime();

	vector<double> x(1);
	if (rank == 0) {
		x.resize(n, 0.0);
	}
	vector<double> myx(rowc[rank]);
	for (int i = 0; i < rowc[rank]; i++) {
		myx[i] = m[i][n];
	}
	COMM_WORLD.Gatherv(&myx[0], rowc[rank], DOUBLE, &x[0], &rowc[0], &frow[0], DOUBLE, 0);
	if (rank == 0) {
		for (int i = 0; i < n; i++) {
			printf("x[%d]=%lf\n", i, x[i]);
		}
		printf("Time: %lf\n", end - start);
	}

	Finalize();
}

