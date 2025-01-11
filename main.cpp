#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include "mpsReader.h"
#include "Simplex.h"

#define pInf numeric_limits<double>::infinity()
#define nInf -numeric_limits<double>::infinity()

int main(int argc, char** argv) {

  std::string filename = argv[1];
  std::string fo = argv[2];
  // int pp = atoi(argv[3]);
  // int refactor = atoi(argv[4]);

  // variaveis para armazenar as informações da instância
  Eigen::MatrixXd A_dense;
  Eigen::VectorXd b;
  Eigen::VectorXd l;
  Eigen::VectorXd u;
  Eigen::VectorXd c;

  int m, n;

  // leitor de instâncias mps
  mpsReader mps;

  if (fo != "mps") {

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
    c = mps.c;
    m = mps.n_rows_eq + mps.n_rows_inq;
    n = mps.n_cols + mps.n_rows_inq + mps.n_rows_eq;

  }

  // Matriz A esparsa
  Eigen::SparseMatrix<double> A = A_dense.sparseView();

  std::cerr << "l:\n" << l << std::endl;
  std::cerr << "\nu:\n" << u << std::endl;
  std::cerr << "\nA_dense:\n" << A_dense << std::endl;
  std::cerr << "\nb:\n" << b << std::endl;
  std::cerr << "\nc:\n" << c << std::endl;
  std::cerr << "\nm: " << m << std::endl;
  std::cerr << "\nn: " << n << std::endl;

  Eigen::MatrixXd A_transpose = A_dense.transpose();
  Data *d = new Data(m, n, A_transpose, c, b, l, u);
  getchar();
  


////////// artificial instance

  int cols = 4, rows = 3;

  std::vector<std::vector<double>> A_in = {
    {3, 1, 4},
    {2, 1, 3},
    {1, 1, 3},
    {2, 1, 4}
  };

  std::vector<double> b_in = {225, 117, 420};
  std::vector<double> c_in = {19, 13, 12, 17};

  Eigen::MatrixXd An = Eigen::MatrixXd(cols, rows);
  for(int i = 0; i < An.innerSize(); i++) {

    for(int j = 0; j < An.row(i).size(); j++) {

      An.row(i)[j] = A_in[i][j];

    }
  } 

  Eigen::VectorXd rhs = Eigen::VectorXd(rows);
  for(int i = 0; i < rows; i++) {
    rhs[i] = b_in[i];
  }

  Eigen::VectorXd cn = Eigen::VectorXd(cols);
  for(int i = 0; i < cols; i++) {
    cn[i] = c_in[i];
  }

  Eigen::VectorXd xn = Eigen::VectorXd(cols);
  for(int i = 0; i < cols; i++) {
    xn[i] = i+1;
  }

  Data *data = new Data(rows, cols, An, xn, cn, rhs, l, u);

  Simplex s = Simplex(data);

  s.solve();

  s.printSolution();

  delete data;
  delete d;

	return 0;

}
