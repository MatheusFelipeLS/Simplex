#include "Data.h"

Data::Data() { 
  /* ctor */
  std::vector<std::vector<double>> A = {
		{3,1,5,6,9,4,3,4,7,6,4,5},
		{1,0,9,5,8,1,2,7,8,7,9,1}
  };

  std::vector<double> b = {72,62};
  std::vector<double> c = {2, 1, -2, -2, 3, 2, 3,-4,0,-2,-3,3};
	std::vector<double> l = {-5,-std::numeric_limits<double>::infinity(),-4,-2,2,0,0,3,-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
	std::vector<double> u = {std::numeric_limits<double>::infinity(),3,-2,3,5,1,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),0,5,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
  std::vector<double> x = {1, 0, -2, 3,2,0, 0, 3, 0, 5, -1, 1};

  this->m = 2;
  this->n = 12;

  std::cout << "1\n";
  getchar();  

  Eigen::MatrixXd sla = Eigen::MatrixXd(m,n);
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      sla(i, j) = A[i][j];
    }
  }

  std::cout << "2\n";
  getchar();  

  this->A = sla.sparseView();

  std::cout << "3\n";
  getchar();  

  this->b = Eigen::VectorXd(m);
  for(int i = 0; i < m; i++) {
    this->b[i] = b[i];
  }

  std::cout << "4\n";
  getchar();  

  this->c = Eigen::VectorXd(n);
  for(int i = 0; i < n; i++) {
    this->c[i] = c[i];
  }

  std::cout << "5\n";
  getchar();  

  this->l = Eigen::VectorXd(n);
  for(int i = 0; i < n; i++) {
    this->l[i] = l[i];
  }

  std::cout << "6\n";
  getchar();  

  this->u = Eigen::VectorXd(n);
  for(int i = 0; i < n; i++) {
    this->u[i] = u[i];
  }

  std::cout << "7\n";
  getchar();  

  this->x = Eigen::VectorXd(n);
  for(int i = 0; i < n; i++) {
    this->x[i] = x[i];
  }

  std::cout << "8\n";
  getchar();  

}


Data::Data(int m, int n, Eigen::VectorXd &c, Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, Eigen::VectorXd &l, Eigen::VectorXd &u) {

  this->m = m;
  this->n = n;
  this->c = c;
  this->A = A;
  this->b = b;
  this->l = l;
  this->u = u;

  this->x = Eigen::VectorXd(n);
  for(int i = 0; i < n-m; i++) x[i] = 0;
  for(int i = 0; i < m; i++) x[i+n-m] = b[i];
}


int Data::qtRows() { return m; }


int Data::qtCols() { return n; }


double Data::getC(int idx) { return c[idx]; }


double Data::getX(int idx) { return x[idx]; }


double Data::getUB(int idx) { return u[idx]; }


double Data::getLB(int idx) { return l[idx]; }


Eigen::VectorXd Data::getCol(int idx) { return A.col(idx); }


Eigen::SparseMatrix<double> Data::getA() { return A; }


void Data::updateX(double t, int idx_ev, Eigen::VectorXd &d, Eigen::VectorXd &B, int signal) {
    
  x[ idx_ev ] += (t * signal);
  
  for(int i = 0; i < m; i++) x[ B[i] ] -= ( (t * d[i]) * signal );

}


double Data::getReducedCost(int idx, Eigen::VectorXd &y) { 
  std::cout << "c[idx]: " << c[idx] << "; y.t: " << y.transpose() << "; A.c: " << A.col(idx).transpose();
  return c[idx] - (y.transpose() * A.col(idx)); 
}


// ?????????????????????
double Data::calculateFO() {

  double value = 0;
  for(int i = 0; i < n-m; i++) {
    value += (x[i] * c[i]);
  }

  return value;

}
