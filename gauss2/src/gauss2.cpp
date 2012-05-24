#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <mpi.h>

using namespace std;
using namespace MPI;

enum ARGS {
	ARG_APPNAME, ARG_SIZE, ARG_C
};

int rank, size;
int n;

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

matrix rows(0, 0);
vector<int> rowid, rowroot;

void gen() {
	int proc = 0;
	rowid.reserve(n / size + 1);
	rowroot.reserve(n);
	for (int i = 0; i < n; i++) {
		if (proc == rank) {
			rowid.push_back(i);
		}
		rowroot.push_back(proc);
		proc = (proc + 1) % size;
	}
	rows = matrix(rowid.size(), n + 1);
	fill(rows[0], rows[rowid.size()], 1.0);
	for (int i = 0; i < rows.h(); i++) {
		rows[i][rowid[i]] = 2.0;
		rows[i][n] = n + 1.0;
		rowid.push_back(i);
	}
}

int main(int argc, char *argv[]) {
	Init(argc, argv);
	rank = COMM_WORLD.Get_rank();
	size = COMM_WORLD.Get_size();

	if (argc != ARG_C) {
		COMM_WORLD.Abort(EXIT_FAILURE);
	}

	n = atoi(argv[ARG_SIZE]);
	if (n <= 0) {
		COMM_WORLD.Abort(EXIT_FAILURE);
	}
	gen();

	COMM_WORLD.Barrier();
	double start = Wtime();
	printf("%d start\n", rank);

	vector<double> row(n + 1);
	for (int i = 0; i < n; i++) {
		int rowoff = i / size;
		if (rowid[rowoff] == i) {
			copy(rows[rowoff] + i, rows[rowoff + 1], row.begin() + i);
		}
		COMM_WORLD.Bcast(&row[i], n + 1 - i, DOUBLE, rowroot[i]);
		for (int j = rowoff; j < rows.h(); j++) {
			if (rowid[j] > i) {
				double a = rows[j][i] / row[i];
				for (int k = i; k < n + 1; k++) {
					rows[j][k] -= a * row[k];
				}
			}
		}
	}

	printf("%d back\n", rank);

	for (int i = n - 1; i >= 0; i--) {
		double v;
		int rowoff = i / size;
		if (rowid[rowoff] == i) {
			v = rows[rowoff][n] / rows[rowoff][i];
		}
		COMM_WORLD.Bcast(&v, 1, DOUBLE, rowroot[i]);
		for (int j = 0; j <= rowoff; j++) {
			if (rowid[j] < i) {
				rows[j][n] -= v * rows[j][i];
			}
		}
	}

	printf("%d done\n", rank);

	COMM_WORLD.Barrier();
	double end = Wtime();

	for (int j = 0; j < rows.h(); j++) {
		rows[j][n] /= rows[j][rowid[j]];
		COMM_WORLD.Isend(&rows[j][n], 1, DOUBLE, 0, rowid[j]);
	}

	if (rank == 0) {
		vector<double> x(n);
		for (int i = 0; i < n; i++) {
			COMM_WORLD.Recv(&x[i], 1, DOUBLE, ANY_SOURCE, i);
			printf("x[%d]=%lf\n", i, x[i]);
		}

		printf("Time: %lf\n", end - start);
	}

	COMM_WORLD.Barrier();
	Finalize();
}

