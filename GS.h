#ifndef GS_H
#define GS_H

#include <iostream>
#include <umfpack.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#define MAX_ETA_SIZE 20

using Eigen::MatrixXd;

class GS {
  public:
    GS();
    GS(int n);

    ~GS();

    void LUDecomposition(int n);

    void BTRAN(Eigen::VectorXd &y);
    
    void FTRAN(Eigen::VectorXd &d, Eigen::VectorXd &a_param);

    void addEtaColumn(int eta_idx, Eigen::VectorXd &eta_column);

    void reinversion();

  private:
    std::vector<std::pair<int, Eigen::VectorXd>> eta;

    Eigen::SparseMatrix<double> B;

    double *null;
		void *Symbolic, *Numeric ;

};

#endif //GS_H