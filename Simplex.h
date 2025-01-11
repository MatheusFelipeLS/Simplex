#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <iostream>
#include <vector>
#include <string>
#include <umfpack.h>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include "GS.h"
#include "Data.h"

#define INFTY 1e9
#define E1 1e-5 // it's about the positivity of reduced cost
#define E2 1e-8 // it's about the choose of entering variable
#define E3 1e-6 // refers to the equivalene Ax* = b 

using Eigen::MatrixXd;

class Simplex {

  public:
  
    Simplex();
    Simplex(Data *d);

    void solve();

    int computeReducedCosts(Eigen::VectorXd &y);

    std::pair<int, double> computeSmallestT(Eigen::VectorXd &d);

    inline double getSolutionValue();

    void update_b(std::pair<int, double> &t, Eigen::VectorXd &d);

    void printSolution();

  private:

    double value; 
    std::string status;

    Data *data;

};

#endif // SIMPLEX_H