#ifndef SCALING_H
#define SCALING_H

#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#define IT 15

class Scaling {

  public:
  
    Scaling();
    ~Scaling();

    void geometric_iterate(Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::VectorXd &c, Eigen::VectorXd &l, Eigen::VectorXd &u);

  private:

    void geometric_scale(Eigen::MatrixXd &A, Eigen::MatrixXd &A_abs, Eigen::VectorXd &b, Eigen::VectorXd &c, Eigen::VectorXd &l, Eigen::VectorXd &u, bool flag);
    std::pair<double, double> compute_min_max_col_ratio(Eigen::MatrixXd &A);
    std::pair<double, double> compute_min_max_row_ratio(Eigen::MatrixXd &A);
    double minCoeff(Eigen::MatrixXd &m);
    double min_in_vector(Eigen::VectorXd &v);

};


#endif
