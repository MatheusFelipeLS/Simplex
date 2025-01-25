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
    std::cout << "mps.n_cols: " << mps.n_cols << "; mps.n_rows_inq: " << mps.n_rows_inq << "; mps.n_rows_eq: " 
    << mps.n_rows_eq << std::endl;

  }

  // Matriz A esparsa
  Eigen::SparseMatrix<double> A = A_dense.sparseView();

  std::cerr << "l:\n" << l.transpose() << std::endl;
  std::cerr << "\nu:\n" << u.transpose() << std::endl;
  std::cerr << "\nA_dense:\n" << A_dense << std::endl;
  std::cerr << "\nb:\n" << b.transpose() << std::endl;
  std::cerr << "\nc:\n" << c.transpose() << std::endl;
  std::cerr << "\nm: " << m << std::endl;
  std::cerr << "\nn: " << n << std::endl;

  std::cerr << "\n3: " << INFTY - 3 << std::endl;
  std::cerr << "\n4: " << INFTY / 4 << std::endl;
  std::cerr << "\n5: " << INFTY + 5 << std::endl;
  std::cerr << "\n6: " << INFTY * 6 << std::endl;
  std::cerr << "\n7: " << (INFTY == INFTY) << std::endl;
  std::cerr << "\n8: " << (INFTY <= INFTY) << std::endl;
  std::cerr << "\n9: " << (INFTY >= INFTY) << std::endl;
  std::cerr << "\n10: " << (INFTY < INFTY) << std::endl;
  std::cerr << "\n11: " << (INFTY > INFTY) << std::endl;
  std::cerr << "\n12: " << ((-1) * (0-(-INFTY))) << std::endl;
  std::cerr << "\n13: " << (INFTY == (INFTY / 4)) << std::endl;


  Data *data = new Data(m, n, c, A, b, l, u);
  getchar();

  Simplex s = Simplex(data);

  s.solve();

  s.printSolution();

  delete data;

	return 0;

}
