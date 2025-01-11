#ifndef DATA_H
#define DATA_H

#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

class Data {

  public:

    Data();

    Data(
      int m, 
      int n,  
      Eigen::MatrixXd &N, 
      Eigen::VectorXd &cn,
      Eigen::VectorXd &b,
      Eigen::VectorXd &l,
      Eigen::VectorXd &u
    );
    
    Data(
      int m, 
      int n,  
      Eigen::MatrixXd &N, 
      Eigen::VectorXd &xn,
      Eigen::VectorXd &cn,
      Eigen::VectorXd &b,
      Eigen::VectorXd &l,
      Eigen::VectorXd &u
    );

    int qtCols();
    int qtRows();

    double getReducedCost(int idx);
    
    double getElement(int i, int j); 
    
    Eigen::SparseMatrix<double> getSparseB();
    Eigen::VectorXd getRow(int row_idx);

    int getXbi(int idx);

    double getbi(int idx);
    void setbi(int idx, double value);
    double getCbi(int idx);
    Eigen::VectorXd getCb();

    void swapBNRow(int b_idx, int n_idx);
    void swapXBNElement(int b_idx, int n_idx);
    void swapCBNElement(int b_idx, int n_idx);


  private:

    int m;
    int n;

    Eigen::MatrixXd N; 
    Eigen::MatrixXd B;

    Eigen::VectorXd xn;
    Eigen::VectorXd xb;
    Eigen::VectorXd x;

    Eigen::VectorXd cb;
    Eigen::VectorXd cn;
    
    Eigen::VectorXd b;

    Eigen::VectorXd l;
    Eigen::VectorXd u;

};

#endif
