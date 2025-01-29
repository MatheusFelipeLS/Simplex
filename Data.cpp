#include "Data.h"

Data::Data() { /* ctor */ }


Data::Data(int m, int n, Eigen::VectorXd &c, Eigen::MatrixXd &A_dense, Eigen::VectorXd &b, Eigen::VectorXd &l, Eigen::VectorXd &u) {

  this->m = m;
  this->n = n;
  this->c = c;
  this->c_aux = c;
  this->b = b;
  this->l = l;
  this->u = u;

  this->l.conservativeResize(n + m);
  this->u.conservativeResize(n + m);
  this->c.conservativeResize(n + m);

  A_dense.conservativeResize(m, m+n);
  for(int i = 0; i < m; i++) A_dense(i, n+i) = 1;

  this->A = A_dense.sparseView();

}


void Data::changeC(bool phase) {

  if(phase) {

    for(int i = 0; i < n-m; i++) c[i] = 0;

  } else {

    for(int i = 0; i < n-m; i++) c[i] = c_aux[i];
    for(int i = n; i < n+m; i++) c[i] = 0;

  }
  
}


int Data::qtRows() { return m; }


int Data::qtCols() { return n; }


double Data::getC(int idx) { return c[idx]; }


double Data::getUB(int idx) { return u[idx]; }


double Data::getLB(int idx) { return l[idx]; }


Eigen::VectorXd Data::getCol(int idx) { return A.col(idx); }


double Data::multiplyByRow(Eigen::VectorXd &x, int idx) {

  double value = 0;
  for(int i = 0; i < n-m; i++) {
    // std::cout << "A.coeffRef(idx, i): " << A.coeffRef(idx, i) << "; x[i]: " << x[i] << std::endl;
    value -= (A.coeffRef(idx, i) * x[i]);
  }
  for(int i = n-m; i < n; i++) {
    // std::cout << "A.coeffRef(idx, i): " << A.coeffRef(idx, i) << "; x[i]: " << x[i] << std::endl;
    value -= (A.coeffRef(idx, i) * x[i]);
  }
  // std::cout << "value: " << value << std::endl;
  return value;

}

Eigen::SparseMatrix<double> Data::getA() { return A; }


double Data::getReducedCost(int idx, Eigen::VectorXd &y) { 
  // std::cout << "c[idx]: " << c[idx] << "; y.t: " << y.transpose() << "; A.c: " << A.col(idx).transpose();
  return c[idx] - (y.transpose() * A.col(idx)); 
}


void Data::setLB(int idx, double value) {
  l[idx] = value;
}


void Data::setUB(int idx, double value) {
  u[idx] = value;
}


void Data::setC(int idx, double value) {
  c[idx] = value;
}


void Data::print() {
  std::cout << "l: " << l.transpose() << "\nu: " << u.transpose() << "\nc: " << c.transpose() << "\n";
}


void Data::resize() {

  l.conservativeResize(n);
  u.conservativeResize(n);
  A.conservativeResize(m, n);

}
