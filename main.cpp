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

  // std::string filename = argv[1];
  // std::string fo = argv[2];
  // int pp = atoi(argv[3]);
  // int refactor = atoi(argv[4]);


  Simplex s = Simplex();

  s.solve();

  s.printSolution();

	return 0;
}
