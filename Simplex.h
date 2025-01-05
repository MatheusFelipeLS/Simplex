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

#define INFTY 1e8 // i suppose that i won't find a value bigger than this
#define E1 1e-5 // it's about the positivity of reduced cost
#define E2 1e-8 // it's about the choose of entering variable
#define E3 1e-6 // refers to the equivalene Ax* = b 

using Eigen::MatrixXd;

class Simplex {

  public:
  
    Simplex();

    void solve();

    int computeReducedCosts(Eigen::VectorXd &y);

    std::pair<int, double> computeSmallestT(Eigen::VectorXd &d);

    inline double getSolutionValue();

    void printSolution();

  private:

    double value; 
    std::string status;

    Eigen::MatrixXd Ab;
    Eigen::MatrixXd An;

    Eigen::VectorXd xb;
    Eigen::VectorXd xn;
    
    Eigen::VectorXd cb;
    Eigen::VectorXd cn;
    
    Eigen::VectorXd b;

};

#endif // SIMPLEX_H