#include "GS.h"

GS::GS() { /* ctor */ }


GS::GS(int n) {
  this->B = Eigen::MatrixXd::Identity(n, n).sparseView() * (-1);

  LUDecomposition(n);

  std::cout << "GS matrix:\n" << B.toDense() << std::endl;
  getchar();
}


GS::GS(Eigen::SparseMatrix<double> &B_param, Eigen::VectorXd &B_, int n) {

  this->B = Eigen::MatrixXd(n, n).sparseView();

  for(int i = 0; i < n; i++) {
    this->B.col(i) = B_param.col( B_[i] );
  }

  std::cout << "GS matrix:\n" << B.toDense() << std::endl;

  LUDecomposition(n);

}


GS::~GS() {
  umfpack_di_free_symbolic(&Symbolic);
  umfpack_di_free_symbolic(&Numeric);
}

// /*

// rapaz, isso parece tá mt mal feito, tqv como as outras pessoas fizeram depois...

void GS::reinversion() {
  
  int n = this->etas_matrix[0].second.size();

  Eigen::MatrixXd B_dense = this->B.toDense();

  for(int i = 0; i < (int) this->etas_matrix.size(); ++i) {
    
    B_dense.row( this->etas_matrix[i].first ) = this->etas_matrix[i].second.transpose() * B_dense;
  
  }

  this->B = B_dense.sparseView();

  this->etas_matrix.clear();

  umfpack_di_free_symbolic (&this->Symbolic);
  umfpack_di_free_numeric (&this->Numeric);

  LUDecomposition(n);
}
// */


void GS::addEtaColumn(int eta_idx, Eigen::VectorXd &eta_column) {

  this->etas_matrix.push_back(std::make_pair(eta_idx, eta_column));

  if(this->etas_matrix.size() == MAX_ETA_SIZE) reinversion();

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

Eigen::VectorXd GS::BTRAN(Eigen::VectorXd &y, Eigen::VectorXd &c_b) {

  // std::cout << "btran\n";
  // std::cout << "y\n" << y << "\nc_b\n" << c_b << "\n";

  for(int i = this->etas_matrix.size()-1; i >= 0; i--) {

    for(int j = 0; j < this->etas_matrix[i].first; j++) {
      y[this->etas_matrix[i].first] -= ( y[j] * this->etas_matrix[i].second[ j ] );
    }

    for(int j = this->etas_matrix[i].first + 1; j < y.size(); j++) {
      y[this->etas_matrix[i].first] -= ( y[j] * this->etas_matrix[i].second[ j ] );
    }
    
    y[this->etas_matrix[i].first] /= this->etas_matrix[i].second[ this->etas_matrix[i].first ];

  }

  // std::cout << "y\n" << y.transpose() << "\nc_b\n" << c_b.transpose() << "\n";
  // resolver v * LU = y, pq tem casos que B != I (equivalente a executar os passos 3, 4, 5 e 6 da BTRAN do livro)
  // /*

  //Aat usa a transposta de B, e B^{T} * y = y^{T} * B^{T} (acho q n precisa especificar q é a transposta de y)
  (void) umfpack_di_solve(UMFPACK_Aat, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), y.data(), c_b.data(), Numeric, null, null);
  // */

  // std::cout << "y\n" << y.transpose() << "\nc_b\n" << c_b.transpose() << "\n";
  getchar();

  return y;

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

Eigen::VectorXd GS::FTRAN(Eigen::VectorXd &d, Eigen::VectorXd &a) {

  // resolver B_0 * d = a (equivalente a executar os passos 1, 2, 3 e 4 da FTRAN do livro)
  // /*  
  (void) umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), d.data(), a.data(), Numeric, null, null);
  // */

  for(long unsigned i = 0; i < this->etas_matrix.size(); i++) {

    d[ this->etas_matrix[i].first ] = (d[ this->etas_matrix[i].first ] / this->etas_matrix[i].second[ this->etas_matrix[i].first ]);

    for(int j = 0; j < this->etas_matrix[i].first; j++) {
      d[ j ] -= (this->etas_matrix[i].second[ j ] * d[ this->etas_matrix[i].first ]);
    }

    for(int j = this->etas_matrix[i].first + 1; j < d.size(); j++) {
      d[ j ] -= (this->etas_matrix[i].second[ j ] * d[ this->etas_matrix[i].first ]);
    }

  }

  return d;

}