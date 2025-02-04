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
  this->A = A_dense.sparseView();

}


void Data::changeObjFunction(bool phase) {

  if(phase) {

    for(int i = 0; i < n-m; i++) c[i] = 0;

  } else {

    for(int i = 0; i < n-m; i++) c[i] = c_aux[i];
    for(int i = n-m; i < n; i++) c[i] = 0;

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
    value += (A.coeffRef(idx, i) * x[i]);
  }

  return value;

}


double Data::getReducedCost(int idx, Eigen::VectorXd &y) { 
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


Eigen::VectorXd Data::copyL() {
  return l;
}


Eigen::VectorXd Data::copyU() {
  return u;
}


Eigen::VectorXd Data::copyC() {
  return c;
}


void Data::print() {
  std::cout << "l: " << l.transpose() << "\nu: " << u.transpose() << "\nc: " << c.transpose() << "\n";
}


void Data::restartLUC(Eigen::VectorXd &l, Eigen::VectorXd &u, Eigen::VectorXd &c) {

  for(int i = n-m; i < n; i++) this->l[i] = l[i];

  for(int i = n-m; i < n; i++) this->u[i] = u[i];

  for(int i = n-m; i < n; i++) this->c[i] = c[i];
  
}



