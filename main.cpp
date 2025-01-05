#include <iostream>
#include <cstdlib>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include <numeric>
#include <ostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>


#include <stdio.h>
#include "Simplex.h"

int main(int argc, char** argv) {

  Simplex s = Simplex();

  s.solve();

  s.printSolution();

	return 0;
}
