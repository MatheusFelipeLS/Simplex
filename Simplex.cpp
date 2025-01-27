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
  this->value = 0;

  this->x = Eigen::VectorXd(data->qtCols());
  for(int i = 0; i < data->qtCols(); i++) x[i] = 0;

  B = Eigen::VectorXd( data->qtRows() );
  for(int i = 0; i < data->qtRows(); i++) B[i] = i + data->qtCols() - data->qtRows();
  
  N = Eigen::VectorXd( data->qtCols() - data->qtRows() );
  for(int i = 0; i < data->qtCols() - data->qtRows(); i++) N[i] = i;

  gs = new GS(data->qtRows());
  
}


Simplex::Simplex(Data *d, Eigen::VectorXd &x) {
  
  this->data = d;
  this->value = 0;

  this->x = x;
  std::cout << "x: " << x.transpose() << "\n" << std::endl;

  B = Eigen::VectorXd( data->qtRows() );
  for(int i = 0; i < data->qtRows(); i++) B[i] = i + data->qtCols() - data->qtRows();
  
  N = Eigen::VectorXd( data->qtCols() - data->qtRows() );
  for(int i = 0; i < data->qtCols() - data->qtRows(); i++) N[i] = i;

  gs = new GS(data->qtRows());
}


Simplex::~Simplex() {
  
}


std::pair<int, int> Simplex::chooseEnteringVariable(Eigen::VectorXd &y) {
  
  double biggest_reduced_cost = 0;
  int idx_biggest = -1;
  int signal = 0;

  for(int i = 0; i < N.size(); i++) {

    double reduced_cost = data->getReducedCost(N[i], y);

    if(x[ N[i] ] < data->getUB(N[i]) && reduced_cost > E1 && reduced_cost > biggest_reduced_cost) {
      signal = 1;
      biggest_reduced_cost = reduced_cost;
      idx_biggest = i;
    }
    
    else if(x[ N[i] ] > data->getLB(N[i]) && reduced_cost < -E1 && std::abs(reduced_cost) > biggest_reduced_cost) {
      signal = -1;
      biggest_reduced_cost = std::abs(reduced_cost);
      idx_biggest = i;
    }

    std::cout << "x[" << N[i] << "]: " << reduced_cost << "; x[N[i]]: " << x[ N[i] ] 
    << "; data->getLB(N[i]): " << data->getLB(N[i]) << "; data->getUB(N[i]): " << 
    data->getUB(N[i]) << std::endl <<std::endl;

  }

  return std::make_pair(idx_biggest, signal);

}


std::pair<int, double> Simplex::chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal) {

  double t;
  double maxt = INFTY;
  int idx_leaving_variable = -1;


  if(signal > 0) maxt = (data->getUB(ent_var) - x[ent_var]);

  else maxt = (x[ent_var] - data->getLB(ent_var));


  for(int i = 0; i < B.size(); i++) {

    double x_b = x[ B[i] ];

    printf("B[i]: %.0f; maxt: %lf; data->getLB( B[i] ): %lf; data->getUB( B[i] ): %lf", 
    B[i], maxt, data->getLB( B[i] ), data->getUB( B[i] ));

    if(std::abs(d[i]) <= E1) {

      continue;
    
    } else if( (signal > 0 && d[i] > 0) || (signal < 0 && d[i] < 0) ) {
      t = signal * (x_b - (data->getLB( B[i] )) ) / d[i];

      if(t >= 0 && t < maxt) {
        maxt = t;
        idx_leaving_variable = i;
      }

    } else if( (signal > 0 && d[i] < 0) || (signal < 0 && d[i] > 0) ) {
      t = signal * (x_b - (data->getUB( B[i] )) ) / d[i];

      if(t >= 0 && t < maxt) {
        maxt = t;
        idx_leaving_variable = i;
      }

    }
    printf("; t: %lf\n", t);
    if(maxt < E1) return std::make_pair(idx_leaving_variable, 0.00);

  }
  
  return std::make_pair(idx_leaving_variable, maxt);

}


void Simplex::updateX(double t, int idx_ev, Eigen::VectorXd &d, int signal) {
    
  x[ idx_ev ] += (t * signal);
  
  for(int i = 0; i < B.size(); i++) x[ B[i] ] -= ( (t * d[i]) * signal );

}


void Simplex::solve() {

  Eigen::VectorXd y(data->qtRows());
  Eigen::VectorXd d(data->qtRows());
  Eigen::VectorXd aux;

  bool newEtaCol = true;

  while(1) {

    if(newEtaCol) {

      for(int i = 0; i < data->qtRows(); i++) y[i] = data->getC(B[i]);

      aux = y;

      std::cout << "\ny: " << y.transpose() << "\naux: " << aux.transpose() << "\n";
      getchar();

      // gs->BTRAN(y, aux);  // solving yB = c
      gs->BTRAN(y);

      std::cout << "y: " << y.transpose() << "\naux: " << aux.transpose() << "\n";
      getchar();

    }

    auto [idx_entering_variable, signal] = chooseEnteringVariable(y);

    std::cout << "\nidx_entering_variable: " << idx_entering_variable << "; signal: " << signal << "\n";
    getchar();

    if(idx_entering_variable == -1) {

      this->status = "Optimal";
      break;

    }

    aux = d = data->getCol( N[ idx_entering_variable ] );

    std::cout << "\nd: " << d.transpose() << "\naux: " << aux.transpose() << "\n";
    getchar();

    gs->FTRAN(d, aux);  // solving Bd = a

    std::cout << "d: " << d.transpose() << "\naux: " << aux.transpose() << "\n";
    getchar();

    auto [idx_leaving_variable, t] = chooseLeavingVariable(d, N[idx_entering_variable], signal);

    std::cout << "\nidx_leaving_variable: " << idx_leaving_variable << "; t: " << t << "\n";
    getchar();

    if(t == INFTY) {

      this->status = "Unbounded";
      this->value = t;
      return;

    }

    std::cout << "x: " << x.transpose() << "\n";
    getchar();

    updateX(t, N[idx_entering_variable], d, signal);  
  
    std::cout << "x: " << x.transpose() << "\n";
    getchar();

    if(idx_leaving_variable > -0.2) {
      
      newEtaCol = true;

      gs->addEtaColumn(idx_leaving_variable, d);

      std::cout << "N: " << N.transpose() << "; B: " << B.transpose() << "\n";

      std::swap(N[ idx_entering_variable ], B[ idx_leaving_variable ]);

      std::cout << "N: " << N.transpose() << "; B: " << B.transpose() << "\n";

    } else {
      newEtaCol = false;
    }

  }

}


void Simplex::printSolution() {

  std::cout << "\nStatus: " << this->status << std::endl; 

  for(int i = 0; i < this->data->qtCols() - this->data->qtRows(); i++) {
    std::cout << "x_" << i+1 << ": " << x[i] << ";  ";
    value += x[i] * data->getC(i);
  }
  std::cout << "\n"; 

  std::cout << "\nOF value: " << this->value << std::endl; 

}


inline double Simplex::getSolutionValue() {
  return this->value;
}
