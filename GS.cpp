#include "GS.h"

GS::GS() {}


GS::GS(Eigen::SparseMatrix<double> &B_param, int n) {

  this->B = B_param;
  LUDecomposition(n);

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
  
  (void) umfpack_di_symbolic (
    n,
    n, 
    this->B.outerIndexPtr(), 
    this->B.innerIndexPtr(), 
    this->B.valuePtr(), 
    &this->Symbolic, 
    this->null, 
    this->null
  );

  (void) umfpack_di_numeric (
    this->B.outerIndexPtr(), 
    this->B.innerIndexPtr(), 
    this->B.valuePtr(), 
    this->Symbolic,
    &this->Numeric, 
    this->null, 
    this->null
  );

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

  for(int i = this->etas_matrix.size()-1; i >= 0; i--) {

    for(int j = 0; j < this->etas_matrix[i].first; j++) {
      y[this->etas_matrix[i].first] -= ( y[j] * this->etas_matrix[i].second[ j ] );
    }

    for(int j = this->etas_matrix[i].first + 1; j < y.size(); j++) {
      y[this->etas_matrix[i].first] -= ( y[j] * this->etas_matrix[i].second[ j ] );
    }
    
    y[this->etas_matrix[i].first] /= this->etas_matrix[i].second[ this->etas_matrix[i].first ];

  }

  // resolver v * LU = y, pq tem casos que B != I (equivalente a executar os passos 3, 4, 5 e 6 da BTRAN do livro)
  // /*
  Eigen::VectorXd y_aux = y; // talvez n seja necessário fazer essa cópia

  (void) umfpack_di_solve(
    UMFPACK_A, 
    this->B.outerIndexPtr(), 
    this->B.innerIndexPtr(), 
    this->B.valuePtr(), 
    y.data(), 
    y_aux.data(), 
    this->Numeric, 
    this->null, 
    this->null
  );
  // */

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
  // lembrando q meu B tá transposto, por isso q eu to usando Aat aqui e não na BTRAN 
  (void) umfpack_di_solve(
    UMFPACK_Aat, //Aat usa a transposta de B, e B^{T} * y = y^{T} * B^{T} (acho q n precisa especificar q é a transposta de y)
    this->B.outerIndexPtr(), 
    this->B.innerIndexPtr(), 
    this->B.valuePtr(), 
    d.data(), 
    a.data(), 
    this->Numeric, 
    this->null, 
    this->null
  );
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