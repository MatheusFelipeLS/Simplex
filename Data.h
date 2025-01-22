#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <limits>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

class Data {

  public:

    Data();

    Data(
      int m, 
      int n,  
      Eigen::VectorXd &c,
      Eigen::SparseMatrix<double> &A,
      Eigen::VectorXd &b,
      Eigen::VectorXd &l,
      Eigen::VectorXd &u
    );

    int qtRows();
    int qtCols();

    double getReducedCost(int idx, Eigen::VectorXd &y);
    double getC(int idx);
    double getX(int idx);
    double getUB(int idx);
    double getLB(int idx);
    Eigen::VectorXd getCol(int idx);

    void updateX(double t, int idx_ev, Eigen::VectorXd &d, Eigen::VectorXd &B, int signal);
    
    Eigen::SparseMatrix<double> getA();

    double calculateFO();

  private:

    int m;
    int n;

    Eigen::VectorXd c;

    Eigen::SparseMatrix<double> A;

    Eigen::VectorXd x;

    Eigen::VectorXd b;

    Eigen::VectorXd l;
    Eigen::VectorXd u;


    //////////////////////// será inútil
    // Eigen::VectorXd xn;
    // Eigen::VectorXd xb;

    // Eigen::VectorXd cb;
    // Eigen::VectorXd cn;

};

#endif // DATA_H
