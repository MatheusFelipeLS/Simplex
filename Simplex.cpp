#include "Simplex.h"

Simplex::~Simplex() { /* dtor */ }


Simplex::Simplex(Data &d, int mes) : data(d) {  
   
  this->value = 0;
  this->MAX_ETA_SIZE = mes;

  this->x = Eigen::VectorXd::Zero(data.qtCols());

  int qtNonBasic = data.qtCols() - data.qtRows();

  N = std::vector<int>(qtNonBasic);
  for(int i = 0; i < qtNonBasic; i++) N[i] = i;

  B = std::vector<int>( data.qtRows() );
  for(int i = 0; i < data.qtRows(); i++) B[i] = i + qtNonBasic;

  gs = new GS(data);
  
}




void Simplex::findInitialSolution() {

  int n = data.qtCols();
  int m = data.qtRows();

  Eigen::VectorXd x_n(n-m);
  Eigen::VectorXd x_b(m);
  Eigen::MatrixXd N_matrix = MatrixXd::Zero(m, n - m);
  for(int i = 0; i < n-m; i++) {
    
    N_matrix.col(i) = data.A.col(i);
    if(data.getUB(i) < INFTY) x_n[i] = data.getUB(i);
    else if(data.getLB(i) > -INFTY) x_n[i] = data.getLB(i);
    else x_n[i] = 0;

  }

  x_b = N_matrix * x_n;

  x << x_n, x_b;

}


bool Simplex::computeInfeasibility() {
  
  int m = data.qtRows();
  int n = data.qtCols();

  double infeasibility = 0;
  for(int i = n-m; i < n; i++) {

    if(x[i] > data.getUB(i)) {

      infeasibility += x[i] - data.getUB(i);
      data.setLB(i, data.getUB(i));
      data.setUB(i, INFTY);
      data.setC(i, -1);

    } else if(x[i] < data.getLB(i)) {

      infeasibility += data.getLB(i) - x[i];
      data.setUB(i, data.getLB(i));
      data.setLB(i, -INFTY);
      data.setC(i, 1);

    }

  }

  // std::cout << "infeasibility: " << infeasibility << std::endl;

  return (infeasibility > E1);
}


std::pair<int, int> Simplex::chooseEnteringVariable(Eigen::VectorXd &y) {
  
  int smallest_idx = std::numeric_limits<int>::max();
  int best_i = -1;
  int signal = 0;

  Eigen::VectorXd reduced_cost = data.getReducedCosts(N, y);

  for(int i = 0; i < (int) N.size(); i++) {

    if(N[i] > smallest_idx) {
      continue;
    }

    if(x[ N[i] ] < data.getUB(N[i]) - E1 && reduced_cost[ N[i] ] > E1) {
      signal = 1;
      smallest_idx = N[i];
      best_i = i;
    }
    
    else if(x[ N[i] ] > data.getLB(N[i]) + E1 && reduced_cost[ N[i] ] < -E1) {
      signal = -1;
      smallest_idx = N[i];
      best_i = i;
    }

  }

  // std::cout << "entering variable: " << N[best_i] << "; signal: " << signal << std::endl;

  return std::make_pair(best_i, signal);

}


std::pair<int, double> Simplex::chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal) {

  double t = 0;
  double maxt = INFTY;
  int idx_leaving_variable = -1;
  int smallest_idx = std::numeric_limits<int>::max();

  // if(signal > 0) maxt = (data.getUB(ent_var) - x[ent_var]);
  // else maxt = (x[ent_var] - data.getLB(ent_var));
  maxt = data.getUB(ent_var) - data.getLB(ent_var);

  for(int i = 0; i < (int) B.size(); i++) {

    if(std::abs(d[i]) <= E1) {

      continue;
    
    } else if( (signal > 0 && d[i] > 0) || (signal < 0 && d[i] < -0) ) {
      t = signal * (x[ B[i] ] - data.getLB( B[i] )) / d[i];
    } else if( (signal > 0 && d[i] < -0) || (signal < 0 && d[i] > 0) ) {
      t = signal * (x[ B[i] ] - data.getUB( B[i] )) / d[i];
    }

    if(t <= maxt) {
      if(t == maxt) {
        if(B[i] < smallest_idx) {
          smallest_idx = B[i];
          idx_leaving_variable = i;
        }
      } else {
        idx_leaving_variable = i;
        smallest_idx = B[i];
      }
      maxt = t;
    }

  }

  // std::cout << "leaving variable: " << B[idx_leaving_variable] << "; t: " << maxt << std::endl;

  return std::make_pair(idx_leaving_variable, maxt);

}


void Simplex::updateX(double t, int idx_ev, Eigen::VectorXd &d, int signal) {
  
  if(t == 0) return;

  x[ idx_ev ] += (t * signal);
  for(int i = 0; i < (int) B.size(); i++) x[ B[i] ] -=  t * d[i] * signal;

}


int Simplex::changeBase(int newEtaCol, Eigen::VectorXd &y) {

  if(newEtaCol) {

    for(int i = 0; i < data.qtRows(); i++) y[i] = data.getC(B[i]);

    gs->BTRAN(y); // solving yB = c

  }

  auto [idx_entering_variable, signal] = chooseEnteringVariable(y);

  if(idx_entering_variable == -1) {

    this->status = "Optimal";
    return 3;

  }

  aux = d = data.getCol( N[ idx_entering_variable ] );

  gs->FTRAN(d, aux);  // solving Bd = a

  auto [idx_leaving_variable, t] = chooseLeavingVariable(d, N[idx_entering_variable], signal);

  if(t == INFTY) {

    this->status = "Unbounded";
    std::cout << "Problem is unbounded\n";
    this->value = t;
    return 2;

  } 

  updateX(t, N[idx_entering_variable], d, signal);  

  if(idx_leaving_variable > -0.1) {
    
    gs->addEtaColumn(idx_leaving_variable, d);

    std::swap(N[ idx_entering_variable ], B[ idx_leaving_variable ]);
    
    if(gs->qtEtaCols() == MAX_ETA_SIZE) gs->reinversion(B);

    return 1;

  } else {
    return 0;
  }

}


void Simplex::simplexLoop(Eigen::VectorXd &y) {

  int newEtaCol = 1;

  int count = 0;
  while(true) {
    count++;
    std::cout << "Iterações: " << count << "\n";

    newEtaCol = changeBase(newEtaCol, y);

    if(newEtaCol > 1) break;

  }

}

void Simplex::solve() {

  Eigen::VectorXd y(data.qtRows());

  data.changeObjFunction(true);

  findInitialSolution(); /* this solution won't be always feasible */
  Eigen::VectorXd l = data.copyL();
  Eigen::VectorXd u = data.copyU();
  Eigen::VectorXd c = data.copyC();

  int newEtaCol = 1;
  while(computeInfeasibility()) {

    newEtaCol = changeBase(newEtaCol, y);

    if(newEtaCol == 3) {
      break;
    }
    if(newEtaCol == 2) {
      status = "Unfeasible";
      return;
    }

    data.restartLUC(l, u, c);

  }

  data.restartLUC(l, u, c);

  data.changeObjFunction(false);

  // std::cout << "PHASE TWO\nx: " << x.transpose() << std::endl;

  simplexLoop(y);

}


void Simplex::printSolution() {

  std::cout << "\nStatus: " << this->status << std::endl; 

  if(status == "Optimal") {
    for(int i = 0; i < this->data.qtCols() - this->data.qtRows(); i++) {
      std::cout << "x_" << i+1 << ": " << x[i] << "; ";
      value += x[i] * data.getC(i);
    }
    std::cout << "\n"; 
  }

  std::cout << "\nObjective value: " << this->value << std::endl; 

}


double Simplex::getSolutionValue() {
  aux = data.copyC();
  return aux.transpose() * x;
  // return this->value;
}
