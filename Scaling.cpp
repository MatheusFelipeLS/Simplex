#include "Scaling.h"

Scaling::Scaling() { /* ctor */ }
Scaling::~Scaling() { /* dtor */ }


std::pair<double, double> Scaling::compute_min_max_row_ratio(Eigen::MatrixXd &A) {

  double min_ratio = std::numeric_limits<double>::infinity();
  double max_ratio = 0;

  for(int i = 0; i < A.rows(); i++) {

    Eigen::VectorXd row = A.row(i);
    double ratio = row.maxCoeff() / min_in_vector(row);

    min_ratio = std::min(ratio, min_ratio);
    max_ratio = std::max(ratio, min_ratio);

  }

  return std::make_pair(min_ratio, max_ratio);

}


std::pair<double, double> Scaling::compute_min_max_col_ratio(Eigen::MatrixXd &A) {

  double min_ratio = std::numeric_limits<double>::infinity();
  double max_ratio = 0;

  for(int i = 0; i < A.cols(); i++) {

    Eigen::VectorXd col = A.col(i);
    double ratio = col.maxCoeff() / col.minCoeff();

    min_ratio = std::min(ratio, min_ratio);
    max_ratio = std::max(ratio, min_ratio);

  }

  return std::make_pair(min_ratio, max_ratio);

}


void Scaling::geometric_scale(Eigen::MatrixXd &A, Eigen::MatrixXd &A_abs, Eigen::VectorXd &b, Eigen::VectorXd &c, Eigen::VectorXd &l, Eigen::VectorXd &u, bool flag) {

  int m = A_abs.rows();
  int n = A.cols();

  std::pair<double, double> min_max;
  double fac;

  if (!flag) {

    for (int j = 0; j < m; j++) {
      
      Eigen::VectorXd row = A_abs.row(j);

      min_max.first = min_in_vector(row);
      min_max.second = row.maxCoeff();

      if (min_max.second == 0) continue;

      fac = 1 / sqrt(min_max.first * min_max.second);
      A.row(j) = A.row(j) * fac;
      b(j) = b(j) * fac;

    }

  } else {

    for (int j = 0; j < n; j++) {

      Eigen::VectorXd col = A_abs.col(j);

      min_max.first = min_in_vector(col);
      min_max.second = col.maxCoeff();      

      if (min_max.second == 0) continue;
      
      double r = sqrt(min_max.first * min_max.second);
      fac = 1 / r;
      A.col(j) = A.col(j) * fac;
      c(j) = c(j) * fac;
      l(j) = l(j) * r;
      u(j) = u(j) * r;

    }

  }

}

double Scaling::minCoeff(Eigen::MatrixXd &m) {

  double minor = std::numeric_limits<double>::infinity();

  for(int i = 0; i < m.rows(); i++) {

    Eigen::VectorXd v = m.row(i);
    double value = min_in_vector(v);

    if(value < minor) minor = value;

  }

  return minor;
}


double Scaling::min_in_vector(Eigen::VectorXd &v) {

  double minor = std::numeric_limits<double>::infinity();

  for(int i = 0; i < v.size(); i++) {

    if(v(i) > 1e-5 && v(i) < minor) minor = v(i);
    
  }

  return minor;
}


void Scaling::geometric_iterate(Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::VectorXd &c, Eigen::VectorXd &l, Eigen::VectorXd &u) {


  Eigen::MatrixXd A_abs = A.cwiseAbs();

  double min_A = minCoeff(A_abs);
  double max_A = A_abs.maxCoeff();

  double old_ratio, ratio = max_A / min_A;

  ratio = 0;

  auto [min_row_ratio, max_row_ratio] = compute_min_max_row_ratio(A_abs);
  auto [min_col_ratio, max_col_ratio] = compute_min_max_col_ratio(A_abs);

  bool flag = ( max_row_ratio > max_col_ratio );

  std::cout << "flag: " << flag << std::endl;

  for(int i = 1; i < IT; i++) {

    std::cout << "iteração: " << i << std::endl;

    old_ratio = ratio;

    geometric_scale(A, A_abs, b, c, l, u, flag);

    A_abs = A.cwiseAbs();

    min_A = minCoeff(A_abs);
    max_A = A_abs.maxCoeff();

    ratio = max_A / min_A;

    if (i > 1 && ratio > 0.9 * old_ratio) break;

  }

}
