#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>

using namespace std;
using namespace MPI;

int rank, size;

double f(double x) {
	return cos(M_PI * x);
}

double parsedbl(const char *s) {
	char *end;
	double v = strtod(s, &end);
	if (*end) {
		if (rank == 0) {
			printf("Invalid parameter %s\n", s);
		}
		COMM_WORLD.Abort(EXIT_FAILURE);
	}
	return v;
}

enum ARGS {
	ARG_APPNAME, ARG_X1, ARG_X2, ARG_STEP, ARG_C
};

int main(int argc, char *argv[]) {
	Init(argc, argv);
	
	rank = COMM_WORLD.Get_rank();
	size = COMM_WORLD.Get_size();
	
	if (argc != ARG_C) {
		printf("Invalid argc\n");
		COMM_WORLD.Abort(EXIT_FAILURE);
	}
	
	double x1 = parsedbl(argv[ARG_X1]);
	double x2 = parsedbl(argv[ARG_X2]);
	double step = parsedbl(argv[ARG_STEP]);
	
	if (x2 < x1) {
		swap(x2, x1);
	}
	if (step <= 0.0) {
		printf("Step must be positive\n");
		COMM_WORLD.Abort(EXIT_FAILURE);
	}
	
	long long steps = (long long)(ceil((x2 - x1) / step));
	
	vector<long long> procsteps(size);
	for (int i = 0; i < size; i++) {
		procsteps[i] = steps / (size - i);
		steps -= procsteps[i];
	}
	
	vector<long long> startstep(size);
	startstep[0] = 0;
	for (int i = 1; i < size; i++) {
		startstep[i] = startstep[i - 1] + procsteps[i - 1];
	}
	
	double z = 0.0;
	#pragma omp parallel for reduction(+:z) schedule(guided)
	for (long long i = 1; i <= procsteps[rank]; i++) {
		double fl = f(x1 + (startstep[rank] + i - 1) * step);
		double fr = f(x2 + (startstep[rank] + i) * step);
		z += (fr + fl) * 0.5 * step;
	}
	
	printf("%d part=%lf\n", rank, z);
	
	double reduced = 0.0;
	COMM_WORLD.Reduce(&z, &reduced, 1, DOUBLE, SUM, 0);
	
	if (rank == 0) {
		printf("result=%lf\n", reduced);
	}
	
	Finalize();
}

