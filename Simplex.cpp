#include "Simplex.h"

Simplex::Simplex() { 
  /* ctor */ 
  B = Eigen::VectorXd( data->qtRows() );
  N = Eigen::VectorXd( data->qtCols() - data->qtRows() );

  B[0] = 0;
  B[1] = 1;

  for(int i = 0; i < data->qtCols() - data->qtRows(); i++) N[i] = i + data->qtRows();
}

Simplex::Simplex(Data *d) {  
  
  this->data = d;  

  B = Eigen::VectorXd( data->qtRows() );
  N = Eigen::VectorXd( data->qtCols() - data->qtRows() );

  for(int i = 0; i < data->qtRows(); i++) B[i] = i + data->qtCols() - data->qtRows();
  
  for(int i = 0; i < data->qtCols() - data->qtRows(); i++) N[i] = i;

  std::cout << "B" << std::endl;
  for(int i = 0; i < data->qtRows(); i++) std::cout << B[i] << " ";
  std::cout << std::endl;
  
  std::cout << "N" << std::endl;
  for(int i = 0; i < data->qtCols() - data->qtRows(); i++) std::cout << N[i] << " ";
  std::cout << std::endl;

  getchar();
  
}


Simplex::~Simplex() {
  delete gs;
}


std::pair<int, int> Simplex::findEnteringVariable(Eigen::VectorXd &y) {
  
  double biggest_reduced_cost = 0;
  int idx_biggest = -1;
  int signal = 0;

  for(int i = 0; i < N.size(); i++) {

    double reduced_cost = data->getReducedCost(N[i], y);

    if(data->getX(N[i]) < data->getUB(N[i]) && reduced_cost > E1 && reduced_cost > biggest_reduced_cost) {
      signal = 1;
      biggest_reduced_cost = reduced_cost;
      idx_biggest = i;
    }
    
    else if(data->getX(N[i]) > data->getLB(N[i]) && reduced_cost < -E1 && std::abs(reduced_cost) > biggest_reduced_cost) {
      signal = -1;
      biggest_reduced_cost = std::abs(reduced_cost);
      idx_biggest = i;
    }

    std::cout << "x[" << N[i] << "]: " << reduced_cost << "; data->getX(N[i]): " << data->getX(N[i]) 
    << "; data->getLB(N[i]): " << data->getLB(N[i]) << "; data->getUB(N[i]): " << data->getUB(N[i]) << std::endl <<std::endl;

  }

  std::cout << "idx_biggest: " << idx_biggest << "; signal: " << signal << std::endl;

  return std::make_pair(idx_biggest, signal);

}


std::pair<int, double> Simplex::chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal) {

  double t;
  double maxt = INFTY;
  int idx_leaving_variable = -1;


  if(signal > 0) maxt = (data->getUB(ent_var) - data->getX(ent_var));

  else maxt = (data->getX(ent_var) - data->getLB(ent_var));


  for(int i = 0; i < B.size(); i++) {

    if(maxt <= E1) return std::make_pair(idx_leaving_variable, 0.00);

    double x = data->getX( B[i] );

      
    if( (signal > 0 && d[i] > 0) || (signal < 0 && d[i] < 0) ) {
      t = signal * (x - (data->getLB( B[i] )) ) / d[i];

      if(t >= 0 && t < maxt) {
        maxt = t;
        idx_leaving_variable = i;
      }

    } else if( (signal > 0 && d[i] < 0) || (signal < 0 && d[i] > 0) ) {
      t = signal * (x - (data->getUB( B[i] )) ) / d[i];

      if(t >= 0 && t < maxt) {
        maxt = t;
        idx_leaving_variable = i;
      }

    }

  }
  
  return std::make_pair(idx_leaving_variable, maxt);

}


void Simplex::solve() {

  Eigen::MatrixXd B_aux = Eigen::MatrixXd::Identity(data->qtRows(), data->qtRows());

  Eigen::SparseMatrix<double> B_param = B_aux.sparseView(); 

  gs = new GS( B_param, data->qtRows() );

  Eigen::VectorXd y(data->qtRows());
  Eigen::VectorXd d(data->qtRows());
  Eigen::VectorXd aux;

  bool newEtaCol = true;

  while(1) {

    if(newEtaCol) {

      for(int i = 0; i < data->qtRows(); i++) y[i] = data->getC(B[i]);

      aux = y;

      gs->BTRAN(y, aux);  // solving yB = c

    }

    auto [idx_entering_variable, signal] = findEnteringVariable(y);

    if(idx_entering_variable == -1) {

      this->status = "Optimal";
      break;

    }

    aux = d = data->getCol( N[ idx_entering_variable ] );

    gs->FTRAN(d, aux);  // solving Bd = a

    auto [idx_leaving_variable, t] = chooseLeavingVariable(d, N[idx_entering_variable], signal);

    if(t == INFTY) {

      this->status = "Unbounded";
      this->value = t;
      return;

    }

    data->updateX(t, N[idx_entering_variable], d, B, signal);  
  
    if(idx_leaving_variable > 0) {

      newEtaCol = true;

      gs->addEtaColumn(idx_leaving_variable, d);

      std::swap(N[ idx_entering_variable ], B[ idx_leaving_variable ]);

    } else {
      newEtaCol = false;
    }

  }

  value = data->calculateFO();

}


void Simplex::printSolution() {

  std::cout << "\nStatus: " << this->status << std::endl; 
  std::cout << "\nOF value: " << this->value << std::endl; 

  for(int i = 0; i < this->data->qtCols() - this->data->qtRows(); i++) 
    std::cout << "x_" << i+1 << ": " << this->data->getX(i) << ";  ";
  std::cout << "\n"; 

}


inline double Simplex::getSolutionValue() {
  return this->value;
}
