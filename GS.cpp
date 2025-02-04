#include "GS.h"

GS::GS() { /* ctor */ }


GS::GS(int n) {

  this->B = Eigen::MatrixXd::Identity(n, n).sparseView() * (-1);

  LUDecomposition(n);

}


GS::~GS() {
  umfpack_di_free_symbolic(&Symbolic);
  umfpack_di_free_symbolic(&Numeric);
}


void GS::reinversion() {
  
  int n = this->eta[0].second.size();

  Eigen::MatrixXd B_dense = this->B.toDense();

  for(int i = 0; i < (int) this->eta.size(); ++i) {
    
    B_dense.col( this->eta[i].first ) = B_dense * this->eta[i].second;
  
  }

  this->B = B_dense.sparseView();

  this->eta.clear();

  umfpack_di_free_symbolic (&this->Symbolic);
  umfpack_di_free_numeric (&this->Numeric);

  LUDecomposition(n);
}


void GS::addEtaColumn(int eta_idx, Eigen::VectorXd &eta_column) {

  this->eta.push_back(std::make_pair(eta_idx, eta_column));

  if(this->eta.size() == MAX_ETA_SIZE) reinversion();

}


void GS::LUDecomposition(int n) {

  this->null = (double *) NULL ;
  
  (void) umfpack_di_symbolic (n,n, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), &Symbolic, null, null);

  (void) umfpack_di_numeric (B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), Symbolic, &Numeric, null, null);

}


// solving yB = c
    // algorithm:
    // y * (B * E_1 * E_2 * ... * E_k) = c

    // Replace (y * B * E_1 * E_2 * ... * E_k-1 ) with v_1
    // Solve v_1 * E_k = c
    // Replace y * B * E_1 * E_2 * ... * E_k-2 with v_2
    // Solve v_2 * E_k-1 = v_1
    // ..
    // Repeat until v_k * B = v_k * LU = v_k-1

void GS::BTRAN(Eigen::VectorXd &y) {

  for(int i = this->eta.size()-1; i >= 0; i--) {

    for(int j = 0; j < this->eta[i].first; j++) {
      y[this->eta[i].first] -= ( y[j] * this->eta[i].second[ j ] );
    }

    for(int j = this->eta[i].first + 1; j < y.size(); j++) {
      y[this->eta[i].first] -= ( y[j] * this->eta[i].second[ j ] );
    }
    
    y[this->eta[i].first] /= this->eta[i].second[ this->eta[i].first ];

  }

  Eigen::VectorXd c_b(y.size());
  for(int i = 0; i < y.size(); i++) c_b[i] = y[i];

  (void) umfpack_di_solve(UMFPACK_Aat, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), y.data(), c_b.data(), Numeric, null, null);

}


// solving Bd = a
// algorithm:
// B * E_1 * E_2 * ... * E_k * d = a

// Replace (E_1 * ... * E_k * d) with v_1
// Solve: B * v_1 = (L * U) * v_1 = a to find v_1
// Now E_1 * E_2 * ... * E_k * d = v_1
// Replace E_2 * ... * E_k * d with v_2
// Solve: E_2 * v_2 = a to find v_2
// Now E_2 * ... * E_k * d = v_2
// repeat until Ek * d = v_k

void GS::FTRAN(Eigen::VectorXd &d, Eigen::VectorXd &a) {

  (void) umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), d.data(), a.data(), Numeric, null, null);

  for(long unsigned i = 0; i < eta.size(); i++) {

    d[ eta[i].first ] = (d[ eta[i].first ] / eta[i].second[ eta[i].first ]);

    for(int j = 0; j < eta[i].first; j++) {
      d[ j ] -= (eta[i].second[ j ] * d[ eta[i].first ]);
    }

    for(int j = eta[i].first + 1; j < d.size(); j++) {
      d[ j ] -= (eta[i].second[ j ] * d[ eta[i].first ]);
    }

  }

}