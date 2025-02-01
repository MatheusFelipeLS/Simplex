#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>
#include <filesystem>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include "mpsReader.h"
#include "Simplex.h"
#include "Data.h"
#include "Scaling.h"

int main(int argc, char** argv) {

  if(argc != 2) {
    std::cout << "Missing parameters. Expected:/\n./solve <instace path>\n";
    exit(0);
  }

  std::string filename = argv[1];
  // std::string fo = argv[2];
  // int pp = atoi(argv[3]);
  // int refactor = atoi(argv[4]);


  std::filesystem::path file(filename);
  std::string extension = file.extension().string();

  // variaveis para armazenar as informações da instância
  Eigen::MatrixXd A_dense;
  Eigen::VectorXd b;
  Eigen::VectorXd l;
  Eigen::VectorXd u;
  Eigen::VectorXd c;

  int m, n;

  // leitor de instâncias mps
  mpsReader mps;

  if (extension != ".mps") {

    ifstream readFile(filename);
    readFile >> m >> n;

    l = VectorXd::Zero(n);
    u = VectorXd::Zero(n);
    A_dense = MatrixXd::Zero(m, n);
    b = VectorXd::Zero(m);
    c = VectorXd::Zero(n);

    readFile.ignore(numeric_limits<streamsize>::max(), '\n');

    for (int i = 0; i < m; i++) {

      for (int j = 0; j < n; j++) {
        readFile >> A_dense(i, j);
      }

      readFile.ignore(numeric_limits<streamsize>::max(), '\n');

    }

    for (int i = 0; i < n; i++) {
      readFile >> c(i);
    }

    readFile.ignore(numeric_limits<streamsize>::max(), '\n');

    std::string str;

    for (int i = 0; i < n; i++) {

      readFile >> str;
      if (!str.compare("inf"))
        l(i) = pInf;
      else if (!str.compare("-inf"))
        l(i) = nInf;
      else
        l(i) = stof(str);
    }

    readFile.ignore(numeric_limits<streamsize>::max(), '\n');

    for (int i = 0; i < n; i++) {
      readFile >> str;
      if (!str.compare("inf"))
        u(i) = pInf;
      else if (!str.compare("-inf"))
        u(i) = nInf;
      else
        u(i) = stof(str);
    }

    readFile.ignore(numeric_limits<streamsize>::max(), '\n');

    for (int i = 0; i < m; i++) {
      readFile >> b(i);
    }

  } else {

    mps.read(filename);

    l = mps.lb;
    u = mps.ub;
    A_dense = mps.A;
    b = mps.b;
    c = -mps.c;
    m = mps.n_rows_eq + mps.n_rows_inq;
    n = mps.n_cols + mps.n_rows_inq + mps.n_rows_eq;
    std::cout << "mps.n_cols: " << mps.n_cols << "; mps.n_rows_inq: " << mps.n_rows_inq << "; mps.n_rows_eq: " 
    << mps.n_rows_eq << std::endl;

  }

  // std::cerr << "\nA_dense:\n" << A_dense << std::endl;
  // std::cerr << "\nc: " << c.transpose() << std::endl;
  // std::cerr << "\nb: " << b.transpose() << std::endl;
  // std::cerr << "\nl: " << l.transpose() << std::endl;
  // std::cerr << "\nu: " << u.transpose() << std::endl;

  // std::cerr << "\n0 == -0: " << (0 == (-0)) << std::endl;
  // std::cerr << "\n0 > -0: " << (0 > (-0)) << std::endl;
  // std::cerr << "\n0 >= -0: " << (0 >= (-0)) << std::endl;
  // getchar();

  std::cerr << "\nm: " << m << std::endl;
  std::cerr << "\nn: " << n << std::endl;

  Data *data = new Data(m, n, c, A_dense, b, l, u);

  Simplex s = Simplex(data);

  s.solve();

  s.printSolution();

  delete data;

	return 0;

}
