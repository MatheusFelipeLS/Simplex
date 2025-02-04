#include "Simplex.h"

Simplex::Simplex() { /* ctor */ }
Simplex::~Simplex() { /* dtor */ }


Simplex::Simplex(Data *d) {  
  
  this->data = d;  
  this->value = 0;
  this->degenerated_iteration = 0;
  this->blands_rule = false;

  this->x = Eigen::VectorXd::Zero(data->qtCols());

  int qtNonBasic = data->qtCols() - data->qtRows();

  N = std::vector<int>(qtNonBasic);
  for(int i = 0; i < qtNonBasic; i++) N[i] = i;

  B = std::vector<int>( data->qtRows() );
  for(int i = 0; i < data->qtRows(); i++) B[i] = i + qtNonBasic;

  gs = new GS(data->qtRows());
  
}


Simplex::Simplex(Data *d, Eigen::VectorXd &x) {
  
  this->data = d;
  this->value = 0;
  this->degenerated_iteration = 0;
  this->blands_rule = false;

  this->x = x;

  B = std::vector<int>( data->qtRows() );
  for(int i = 0; i < data->qtRows(); i++) B[i] = i + data->qtCols() - data->qtRows();
  
  N = std::vector<int>( data->qtCols() - data->qtRows() );
  for(int i = 0; i < data->qtCols() - data->qtRows(); i++) N[i] = i;

  gs = new GS(data->qtRows());
}


std::pair<int, int> Simplex::chooseEnteringVariable(Eigen::VectorXd &y) {
  
  // double biggest_reduced_cost = 0;
  int biggest_reduced_cost = std::numeric_limits<int>::max();
  int idx_biggest = -1;
  int signal = 0;

  for(int i = 0; i < (int) N.size(); i++) {

    double reduced_cost = data->getReducedCost(N[i], y);

    if(x[ N[i] ] < data->getUB(N[i]) - E1 && reduced_cost > E1 && N[i] < biggest_reduced_cost) {
    // if(x[ N[i] ] < data->getUB(N[i]) - E1 && reduced_cost > E1 && reduced_cost > biggest_reduced_cost) {

      // biggest_reduced_cost = reduced_cost;
      signal = 1;
      biggest_reduced_cost = N[i];
      idx_biggest = i;
      
    }
    
    else if(x[ N[i] ] > data->getLB(N[i]) + E1 && reduced_cost < -E1 && N[i] < biggest_reduced_cost) {
    // else if(x[ N[i] ] > data->getLB(N[i]) + E1 && reduced_cost < -E1 && std::abs(reduced_cost) > biggest_reduced_cost) {

      // biggest_reduced_cost = std::abs(reduced_cost);
      signal = -1;
      biggest_reduced_cost = N[i];
      idx_biggest = i;

    }

  }

  return std::make_pair(idx_biggest, signal);

}


std::pair<int, double> Simplex::chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal) {

  double t;
  double maxt = INFTY;
  int idx_leaving_variable = -1;


  if(signal > 0) maxt = (data->getUB(ent_var) - x[ent_var]);

  else maxt = (x[ent_var] - data->getLB(ent_var));


  for(int i = 0; i < (int) B.size(); i++) {

    double x_b = x[ B[i] ];

    if(std::abs(d[i]) <= E2) {

      continue;
    
    } else if( (signal > 0 && d[i] > 0) || (signal < 0 && d[i] < 0) ) {

      t = signal * (x_b - (data->getLB( B[i] )) ) / d[i];
      if((t >= 0 && t < maxt) ) {
        maxt = t;
        idx_leaving_variable = i;
      }

    } else if( (signal > 0 && d[i] < 0) || (signal < 0 && d[i] > 0) ) {
      
      t = signal * (x_b - (data->getUB( B[i] )) ) / d[i];
      if( (t >= 0 && t < maxt) ) {
        maxt = t;
        idx_leaving_variable = i;
      }

    }

  }
  
  return std::make_pair(idx_leaving_variable, maxt);

}


void Simplex::updateX(double t, int idx_ev, Eigen::VectorXd &d, int signal) {
    
  x[ idx_ev ] += (t * signal);
  
  for(int i = 0; i < (int) B.size(); i++) x[ B[i] ] -= ( (t * d[i]) * signal );

}


void Simplex::simplexLoop(Eigen::VectorXd &y) {

  int newEtaCol = 1;

  int count = 0;
  while(true) {
    count++;
    std::cout << "Iterações: " << count << "\n";

    newEtaCol = Maximize(newEtaCol, y);

    if(newEtaCol > 1) {
      break;
    }

  }

}


int Simplex::Maximize(int newEtaCol, Eigen::VectorXd &y) {

  Eigen::VectorXd d(data->qtRows());
  Eigen::VectorXd aux;

  if(newEtaCol) {

    for(int i = 0; i < data->qtRows(); i++) y[i] = data->getC(B[i]);

    gs->BTRAN(y); // solving yB = c

  }

  auto [idx_entering_variable, signal] = chooseEnteringVariable(y);

  if(idx_entering_variable == -1) {

    this->status = "Optimal";
    return 3;

  }

  aux = d = data->getCol( N[ idx_entering_variable ] );

  gs->FTRAN(d, aux);  // solving Bd = a

  auto [idx_leaving_variable, t] = chooseLeavingVariable(d, N[idx_entering_variable], signal);

  if(t == INFTY) {

    this->status = "Unbounded";
    std::cout << "Problem is unbounded\n";
    this->value = t;
    return 2;

  } else if(t < E1) {

    degenerated_iteration++;

    if(degenerated_iteration == MAX_DEGENERATED_ITERATION) {
      blands_rule = true;
    }

  } else {
    
    blands_rule = false;
    degenerated_iteration = 0;
    
  }

  updateX(t, N[idx_entering_variable], d, signal);  

  double rowSum;
  for(int i = 0; i < data->qtRows(); i++) {

    rowSum = data->multiplyByRow(x, i) - x[i+data->qtCols()-data->qtRows()];

    if(std::abs(rowSum) > E3) {

      gs->reinversion();
      break;

    }

  }


  if(idx_leaving_variable > -0.1) {
    
    gs->addEtaColumn(idx_leaving_variable, d);

    std::swap(N[ idx_entering_variable ], B[ idx_leaving_variable ]);

    return 1;

  } else {

    return 0;
    
  }

}


void Simplex::solve() {

  Eigen::VectorXd y(data->qtRows());

  data->changeObjFunction(true);

  findInitialSolution(); /* this solution won't be always feasible */

  Eigen::VectorXd l = data->copyL();
  Eigen::VectorXd u = data->copyU();
  Eigen::VectorXd c = data->copyC();


  int newEtaCol = 1;
  while(computeInfeasibility()) {

    newEtaCol = Maximize(newEtaCol, y);

    if(newEtaCol == 2) {
      status = "Unfeasible";
      return;
    }

    data->restartLUC(l, u, c);

  }

  data->restartLUC(l, u, c);

  data->changeObjFunction(false);


  std::cout << "PHASE TWO\nx: " << x.transpose() << std::endl;

  simplexLoop(y);

}


void Simplex::findInitialSolution() {

  int n = data->qtCols();
  int m = data->qtRows();

  Eigen::VectorXd x_n(n-m);
  Eigen::VectorXd x_b(m);

  for(int i = 0; i < n-m; i++) {
    
    if(data->getLB(i) > -INFTY) x_n[i] = data->getLB(i);
    else if(data->getUB(i) < INFTY) x_n[i] = data->getUB(i);
    else x_n[i] = 0;

  }


  for(int i = 0; i < m; i++) {
    x_b(i) = data->multiplyByRow(x_n, i);
  }

  x << x_n, x_b;

}


bool Simplex::computeInfeasibility() {
  
  int m = data->qtRows();
  int n = data->qtCols();

  double infeasibility = 0;
  for(int i = n-m; i < n; i++) {

    if(x[i] > data->getUB(i)) {

      infeasibility += x[i] - data->getUB(i);
      data->setLB(i, data->getUB(i));
      data->setUB(i, INFTY);
      data->setC(i, -1);

    } else if(x[i] < data->getLB(i)) {

      infeasibility += data->getLB(i) - x[i];
      data->setUB(i, data->getLB(i));
      data->setLB(i, -INFTY);
      data->setC(i, 1);

    }

  }

  std::cout << "infeasibility: " << infeasibility << std::endl;



  return (infeasibility > E1);
}


void Simplex::printSolution() {

  std::cout << "\nStatus: " << this->status << std::endl; 

  if(status == "Optimal") {
    for(int i = 0; i < this->data->qtCols() - this->data->qtRows(); i++) {
      std::cout << "x_" << i+1 << ": " << x[i] << ";  ";
      value += x[i] * data->getC(i);
    }
    std::cout << "\n"; 
  }

  std::cout << "\nObjective value: " << this->value << std::endl; 

}


inline double Simplex::getSolutionValue() {
  return this->value;
}
