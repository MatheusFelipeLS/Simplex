#include "Data.h"

Data::Data() { /* ctor */}

Data::Data(int m, int n, Eigen::MatrixXd &N, Eigen::VectorXd &xn, Eigen::VectorXd &cn, Eigen::VectorXd &b) {
  this->m = m;
  this->n = n;
  this->N = N;
  this->xn = xn;
  this->cn = cn;
  this->b = b;

  this->B = Eigen::MatrixXd::Identity(this->m, this->m);

  this->cb = Eigen::VectorXd( this->m );
  this->cb.setZero();

  this->xb = Eigen::VectorXd( this->m );
  for(int i = 0; i < m; i++) {
    this->xb[i] = i+n+1;
  }

}


int Data::qtRows() { return this->m; }


int Data::qtCols() { return this->n; }


void Data::swapBNRow(int b_idx, int n_idx) {
  this->N.row(n_idx).swap(this->B.row(b_idx));
}


void Data::swapXBNElement(int b_idx, int n_idx) {
  std::swap(this->xb[b_idx], this->xn[n_idx]);
}


void Data::swapCBNElement(int b_idx, int n_idx) {
  std::swap(this->cb[b_idx], this->cn[n_idx]);
}


double Data::getReducedCost(int idx) {
  return this->cn[idx];
}


double Data::getElement(int i, int j) {
  return this->N(i, j);
}


Eigen::VectorXd Data::getRow(int row_idx) {
  return this->N.row(row_idx);
}


double Data::getbi(int idx) {
  return this->b[idx];
}


void Data::setbi(int idx, double value) {
  this->b[idx] = value;
}


double Data::getCbi(int idx) {
  return this->cb[idx];
}


Eigen::VectorXd Data::getCb() {
  return this->cb;
}


int Data::getXbi(int idx) {
  return this->xb[idx];
}
