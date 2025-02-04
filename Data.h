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
      Eigen::MatrixXd &A_dense,
      Eigen::VectorXd &b,
      Eigen::VectorXd &l,
      Eigen::VectorXd &u
    );

    int qtRows();
    int qtCols();

    void changeObjFunction(bool phase);
    Eigen::VectorXd copyL();
    Eigen::VectorXd copyU();
    Eigen::VectorXd copyC();

    Eigen::VectorXd getCol(int idx);
    double getReducedCost(int idx, Eigen::VectorXd &y);
    double getC(int idx);
    double getX(int idx);
    double getUB(int idx);
    double getLB(int idx);
    
    double multiplyByRow(Eigen::VectorXd &x, int idx);
    

    void setLB(int idx, double value);
    void setUB(int idx, double value);
    void setC(int idx, double value);

    void restartLUC(Eigen::VectorXd &l, Eigen::VectorXd &u, Eigen::VectorXd &c);

    void print();

  private:

    int m;
    int n;

    Eigen::VectorXd c;
    Eigen::VectorXd c_aux;

    Eigen::SparseMatrix<double> A;

    Eigen::VectorXd b;

    Eigen::VectorXd l;
    Eigen::VectorXd u;

};

#endif // DATA_H
