#include "Simplex.h"

Simplex::Simplex() { /* ctor */ }

Simplex::Simplex(Data *d) {  this->data = d;  }


inline double Simplex::getSolutionValue() {
  return this->value;
}


int Simplex::computeReducedCosts(Eigen::VectorXd &y) {
  
  double biggest_reduced_cost = -INFTY;
  int idx_biggest = 0;

  for(int i = 0; i < this->data->qtCols(); i++) {

    double reduced_cost = this->data->getReducedCost(i);

    for(int j = 0; j < y.size(); j++) reduced_cost -= y[j] * this->data->getElement(i, j);

    if(reduced_cost > 0 && this->data->getXbi()) {

      biggest_reduced_cost = reduced_cost;
      idx_biggest = i;
      
    } else if(reduced_cost < 0 && this->data->getXbi()) {

      

    }

    std::cout << "x[" << i+1 << "]: " << reduced_cost << std::endl;

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

    if( (d[i] >= 0 && this->data->getbi(i) < 0) || (d[i] <= 0 && this->data->getbi(i) > 0) ) continue;

    if(d[i]) t = this->data->getbi(i) / d[i];
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

void Simplex::update_b(std::pair<int, double> &t, Eigen::VectorXd &d) {

  for(int i = 0; i < this->data->qtRows(); i++) {

    if(i != t.first) this->data->setbi(i, this->data->getbi(i) - (t.second * d[i]));
    else this->data->setbi(i, t.second);
  
  }
}


void Simplex::solve() {

  Eigen::SparseMatrix<double> B_param = this->data->getSparseB();
  GS gs = GS(B_param, this->data->qtRows() );

  Eigen::VectorXd y;
  Eigen::VectorXd d;
  Eigen::VectorXd aux;

  while(1) {

    // y = this->cb;
    y = aux = this->data->getCb();


    // solving yB = c
    gs.BTRAN(y, aux);
    

    int choosen = computeReducedCosts(y);
    std::cout << "idx reduced: " << choosen << std::endl;

    if(choosen == -1) {

      this->status = "Optimal";
      break;

    }

    
    // std::cout << "Choose the variable to enter the basis (it reduced cost must be positive): ";
    // std::cin >> choosen;
    // choosen--;

    // Eigen::VectorXd aux = d = this->An.row(choosen);
    aux = d = this->data->getRow(choosen);
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

    
    update_b(t, d);


    this->data->swapBNRow(t.first, choosen);
    this->data->swapXBNElement(t.first, choosen);
    this->data->swapCBNElement(t.first, choosen);

  }

  this->value = 0;
  for(int i = 0; i < this->data->qtRows(); i++) this->value += ( this->data->getbi(i) * this->data->getCbi(i) );

}


void Simplex::printSolution() {
  std::cout << "\nOptimal: " << this->value << std::endl; 

  for(int i = 0; i < this->data->qtRows(); i++) 
    std::cout << "x_" << this->data->getXbi(i) << ": " << this->data->getbi(i) << ";  ";
  std::cout << "\n"; 

}
