#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <umfpack.h>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include "GS.h"
#include "Data.h"

#define INFTY std::numeric_limits<double>::infinity()
#define E1 1e-5 // it's about the positivity of reduced cost
#define E2 1e-8 // it's about the choose of entering variable
#define E3 1e-6 // refers to the equivalene Ax* = b 

using Eigen::MatrixXd;

class Simplex {

  public:
  
    Simplex();
    Simplex(Data *d);
    Simplex(Data *d, Eigen::VectorXd &x);
    ~Simplex();

    void solve();

    std::pair<int, int> chooseEnteringVariable(Eigen::VectorXd &y);
    
    std::pair<int, double> chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal);

    inline double getSolutionValue();

    void updateX(double t, int idx_ev, Eigen::VectorXd &d, int signal);

    void printSolution();

    int FirstPhase();

  private:

    double value; 
    std::string status;

    Data *data;

    GS *gs;

    Eigen::VectorXd x;
    Eigen::VectorXd B;
    Eigen::VectorXd N; 

};

#endif // SIMPLEX_H