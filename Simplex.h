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
#define E3 1e-6 // Ax* = b 
#define MAX_DEGENERATED_ITERATION 50

using Eigen::MatrixXd;

class Simplex {

  public:
  
    Simplex(Data &d, int mes); // mes = MAX ETA SIZE
    ~Simplex();

    void findInitialSolution();
    bool computeInfeasibility();

    void solve();
    int changeBase(int newEtaCol, Eigen::VectorXd &y);
    void simplexLoop(Eigen::VectorXd &y);

    std::pair<int, int> chooseEnteringVariable(Eigen::VectorXd &y);
    std::pair<int, double> chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal);
    void updateX(double t, int idx_ev, Eigen::VectorXd &d, int signal);
    
    void printSolution();
    double getSolutionValue();

  private:

    double value; 
    std::string status;

    Data &data;

    GS *gs;

    Eigen::VectorXd x;
    std::vector<int> B;
    std::vector<int> N; 

    Eigen::VectorXd d;
    Eigen::VectorXd aux;

    int MAX_ETA_SIZE;

};

#endif // SIMPLEX_H