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
    Eigen::VectorXd getReducedCosts(std::vector<int> &N, Eigen::VectorXd &y);
    double getC(int idx);
    double getUB(int idx);
    double getLB(int idx);
    Eigen::SparseMatrix<double> getA();
    
    double multiplyByRow(Eigen::VectorXd &x, int idx);
    

    void setLB(int idx, double value);
    void setUB(int idx, double value);
    void setC(int idx, double value);

    void restartLUC(Eigen::VectorXd &l, Eigen::VectorXd &u, Eigen::VectorXd &c);

    void print();
    void printC();
    void printA();

    Eigen::SparseMatrix<double> A;
  private:

    int m;
    int n;

    Eigen::VectorXd c;
    Eigen::VectorXd c_aux;


    Eigen::VectorXd b;

    Eigen::VectorXd l;
    Eigen::VectorXd u;

};

#endif // DATA_H
