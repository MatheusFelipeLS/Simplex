#include "Simplex.h"

Simplex::Simplex() { /* ctor */ }


inline double Simplex::getSolutionValue() {
  return this->value;
}


void Simplex::printSolution() {
  std::cout << "\nOptimal: " << this->value << std::endl; 
  for(int i = 0; i < this->xb.size(); i++) std::cout << "x_" << this->xb[i]+1 << ": " << this->b[i] << ";  ";
  std::cout << "\n"; 
}


int Simplex::computeReducedCosts(Eigen::VectorXd &y) {
  
  double biggest_reduced_cost = -INFTY;
  int idx_biggest = 0;

  for(int i = 0; i < this->An.rows(); i++) {

    double reduced_cost = this->cn[i];

    for(int j = 0; j < y.size(); j++) reduced_cost -= y[j] * this->An(i, j);

    if(reduced_cost > biggest_reduced_cost) {

      biggest_reduced_cost = reduced_cost;
      idx_biggest = i;

    }

    std::cout << "x[" << i << "]: " << reduced_cost << std::endl;

  }

  if(biggest_reduced_cost < E1) return -1;

  return idx_biggest;

}


std::pair<int, double> Simplex::computeSmallestT(Eigen::VectorXd &d) {

  double small_T = INFTY;
  int idx_small_T = 0;
  bool unbounded = true;
  double t;
  
  for(int i = 0; i < d.size(); i++) {

    if( (d[i] >= 0 && this->b[i] < 0) || (d[i] <= 0 && this->b[i] > 0) ) continue;

    if(d[i]) t = this->b[i] / d[i];
    else t = INFTY;
    
    unbounded = false;

    if(t < small_T) {
      small_T = t;
      idx_small_T = i;
    }

  }

  if(unbounded) return std::make_pair(-1, 0);
  
  return std::make_pair(idx_small_T, small_T);

}


void Simplex::solve() {

  int n = 4, m = 3;

  std::vector<std::vector<double>> A_in = {
    {3, 1, 4},
    {2, 1, 3},
    {1, 1, 3},
    {2, 1, 4}
  };

  std::vector<double> b_in = {225, 117, 420};
  std::vector<double> c_in = {19, 13, 12, 17};
  std::vector<int> x_b_in = {4, 5, 6};


  this->An = Eigen::MatrixXd(n, m);
  for(int i = 0; i < this->An.innerSize(); i++) {

    for(int j = 0; j < this->An.row(i).size(); j++) {

      this->An.row(i)[j] = A_in[i][j];

    }
  } 

  this->Ab = Eigen::MatrixXd::Identity(m,m);

  this->b = Eigen::VectorXd(m);
  for(int i = 0; i < m; i++) {
    this->b[i] = b_in[i];
  }

  this->cn = Eigen::VectorXd(n);
  for(int i = 0; i < n; i++) {
    this->cn[i] = c_in[i];
  }

  this->cb = Eigen::VectorXd(m);
  cb.setZero();

  this->xb = Eigen::VectorXd(m);
  for(int i = 0; i < m; i++) {
    this->xb[i] = x_b_in[i];
  }


  GS gs = GS();

  Eigen::VectorXd y(m);
  Eigen::VectorXd d(m);

  while(1) {

    for(int i = 0; i < this->cb.size(); i++) y[ i ] = this->cb[ i ];

    // solving yB = c
    gs.BTRAN(y, this->cb);
    

    int choosen = computeReducedCosts(y);
    std::cout << "idx reduced: " << choosen << std::endl;

    if(choosen == -1) {

      this->status = "Optimal";
      break;

    }

    
    std::cout << "Choose the variable to enter the basis (it reduced cost must be positive): ";
    std::cin >> choosen;

    Eigen::VectorXd aux = this->An.row(choosen);

    d = aux;

    gs.FTRAN(d, aux);

    std::cout << "aux:\n" << aux << std::endl;
    std::cout << "d:\n" << d << std::endl;


    auto t = computeSmallestT(d);

    if(t.first == -1) {

      this->status = "Unbounded";
      this->value = INFTY;
      return;

    }

    std::cout << "value: " << t.second << "; idx: " << t.first << std::endl;

    gs.addEtaColumn(t.first, d);

    for(int i = 0; i < b.size(); i++) {
      if(i != t.first) this->b[i] -= (t.second * d[i]);
      else this->b[i] = t.second;
    }



    this->xb[t.first] = choosen;
    int aux_reduced_cost = this->cb[t.first];
    this->cb[t.first] = this->cn[choosen];
    this->cn[choosen] = aux_reduced_cost;
    this->An.row(choosen) = this->Ab.row(t.first);

  }

  this->value = 0;
  for(int i = 0; i < b.size(); i++) this->value += ( this->b[i] * this->cb[ i ] );

}

