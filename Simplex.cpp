#include "Simplex.h"

Simplex::Simplex() { /* ctor */ }


Simplex::Simplex(Data *d) {  
  
  this->data = d;  
  this->value = 0;
  this->degenerated_iteration = 0;
  this->blands_rule = false;

  this->x = Eigen::VectorXd(data->qtCols() + data->qtRows());
  for(int i = 0; i < data->qtCols() + data->qtRows(); i++) x[i] = 0;

  B = Eigen::VectorXd( data->qtRows() );
  for(int i = 0; i < data->qtRows(); i++) B[i] = i + data->qtCols();
  
  N = Eigen::VectorXd(data->qtCols());
  for(int i = 0; i < data->qtCols(); i++) N[i] = i;

  gs = new GS(data->qtRows());
  
}


Simplex::Simplex(Data *d, Eigen::VectorXd &x) {
  
  this->data = d;
  this->value = 0;
  this->degenerated_iteration = 0;
  this->blands_rule = false;

  this->x = x;
  // std::cout << "x: " << x.transpose() << "\n" << std::endl;

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

    if(x[ N[i] ] < data->getUB(N[i]) - E1 && reduced_cost > E1 && reduced_cost > biggest_reduced_cost) {

      signal = 1;
      biggest_reduced_cost = reduced_cost;
      idx_biggest = i;

      if(blands_rule) return std::make_pair(idx_biggest, signal);
      
    }
    
    else if(x[ N[i] ] > data->getLB(N[i]) + E1 && reduced_cost < -E1 && std::abs(reduced_cost) > biggest_reduced_cost) {

      signal = -1;
      biggest_reduced_cost = std::abs(reduced_cost);
      idx_biggest = i;

      if(blands_rule) return std::make_pair(idx_biggest, signal);

    }

    // std::cout << "x[" << N[i] << "]: " << reduced_cost << "; x[N[i]]: " << x[ N[i] ] 
    // << "; data->getLB(N[i]): " << data->getLB(N[i]) << "; data->getUB(N[i]): " << 
    // data->getUB(N[i]) << std::endl <<std::endl;

  }

  return std::make_pair(idx_biggest, signal);

}


std::pair<int, double> Simplex::chooseLeavingVariable(Eigen::VectorXd &d, int ent_var, int signal) {

  double t;
  double maxt = INFTY;
  int idx_leaving_variable = -1;


  if(signal > 0) maxt = (data->getUB(ent_var) - x[ent_var]);

  else maxt = (x[ent_var] - data->getLB(ent_var));

  // std::cout << "data->getUB(ent_var): " << data->getUB(ent_var) << "; data->getLB(ent_var): " << data->getLB(ent_var) << "; x[ent_var]: " << x[ent_var] << std::endl;

  for(int i = 0; i < B.size(); i++) {

    double x_b = x[ B[i] ];

    if(std::abs(d[i]) <= E2) {

      // std::cout << "d[" << i << "]: " << d[i] << "; B[i]: " << B[i] << "; idx_leaving_variable: " << idx_leaving_variable << "; maxt: " << maxt << std::endl;
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

    // std::cout <<"B[" << i << "]: " << B[i] << "; maxt: " << maxt << "; idx_lv: "
    // << idx_leaving_variable << "; data->getLB( B[i] ): " << data->getLB( B[i] ) 
    // << "; data->getUB( B[i] ): " << data->getUB( B[i] ) << "; t: " << t 
    // << "; x_b: " << x_b << "\n";

    if(maxt < E1) return std::make_pair(idx_leaving_variable, 0.000000);

  }
  
  return std::make_pair(idx_leaving_variable, maxt);

}


void Simplex::sortLists(int idx_entering_variable, int idx_leaving_variable) {
  
  // std::cout <<"idx_entering_variable: " << idx_entering_variable << "; idx_leaving_variable: " << idx_leaving_variable << "; B.size(): " << B.size() << "; N.size(): " << N.size() << std:: endl;

  while(idx_entering_variable < N.size()-1) { 

    if(N[ idx_entering_variable ] > N[ idx_entering_variable+1 ]) {
      std::swap(N[ idx_entering_variable ], N[ idx_entering_variable+1 ]);
      idx_entering_variable++;
    } else break;

  }

  while(idx_entering_variable > 0) {

    if(N[ idx_entering_variable ] < N[ idx_entering_variable-1 ]) {
      std::swap(N[ idx_entering_variable ], N[ idx_entering_variable-1 ]);
      idx_entering_variable--;
    } else break;

  }

  while(idx_leaving_variable < B.size()-1) {

    if(B[ idx_leaving_variable ] > B[ idx_leaving_variable+1 ]) {
      std::swap(B[ idx_leaving_variable ], B[ idx_leaving_variable+1 ]);
      idx_leaving_variable++;
    } else break;

  }

  while(idx_leaving_variable > 0) {

    if(B[ idx_leaving_variable ] < B[ idx_leaving_variable-1 ]) {
      std::swap(B[ idx_leaving_variable ], B[ idx_leaving_variable-1 ]);
      idx_leaving_variable--;
    } else break;

  }

}


void Simplex::updateX(double t, int idx_ev, Eigen::VectorXd &d, int signal) {
    
  x[ idx_ev ] += (t * signal);
  
  for(int i = 0; i < B.size(); i++) x[ B[i] ] -= ( (t * d[i]) * signal );

}


void Simplex::Maximize() {

  Eigen::VectorXd y(data->qtRows());
  Eigen::VectorXd d(data->qtRows());
  Eigen::VectorXd aux;

  bool newEtaCol = true;

  int count = 0;
  while(true) {
    count++;
    std::cout << count << std::endl;

    if(newEtaCol) {

      for(int i = 0; i < data->qtRows(); i++) y[i] = data->getC(B[i]);

      // std::cout << "\ny: " << y.transpose() << "\naux: " << aux.transpose() << "\n";
      // getchar();

      // gs->BTRAN(y, aux);  // solving yB = c
      gs->BTRAN(y);

      // std::cout << "y: " << y.transpose() << "\naux: " << aux.transpose() << "\n";
      // getchar();

    }

    auto [idx_entering_variable, signal] = chooseEnteringVariable(y);

    // std::cout << "\nidx_entering_variable: " << idx_entering_variable << "; signal: " << signal << "\n";
    // getchar();

    if(idx_entering_variable == -1) {

      this->status = "Optimal";
      break;

    }

    aux = d = data->getCol( N[ idx_entering_variable ] );

    // std::cout << "\nd: " << d.transpose() << "\naux: " << aux.transpose() << "\n";
    // getchar();

    gs->FTRAN(d, aux);  // solving Bd = a

    // std::cout << "d: " << d.transpose() << "\naux: " << aux.transpose() << "\n";
    // getchar();

    auto [idx_leaving_variable, t] = chooseLeavingVariable(d, N[idx_entering_variable], signal);

    // std::cout << "\nidx_leaving_variable: " << idx_leaving_variable << "; t: " << t << "\n";
    // getchar();

    if(t == INFTY) {

      this->status = "Unbounded";
      std::cout << "Unbounded\n";
      this->value = t;
      return;

    } 
    // /*
    else if(t < E1) {

      degenerated_iteration++;

      if(degenerated_iteration == MAX_DEGENERATED_ITERATION) {
        blands_rule = true;
      }

    } else {
      
      blands_rule = false;
      degenerated_iteration = 0;
      
    }
    // */

    // std::cout << "x: " << x.transpose() << "\n";
    // getchar();

    updateX(t, N[idx_entering_variable], d, signal);  
  
    // std::cout << "x: " << x.transpose() << "\n";
    // getchar();

    if(idx_leaving_variable > -0.1) {
      
      newEtaCol = true;

      gs->addEtaColumn(idx_leaving_variable, d);

      // std::cout << "N: " << N.transpose() << "\n\nB: " << B.transpose() << "\n\n\n";

      std::swap(N[ idx_entering_variable ], B[ idx_leaving_variable ]);

/*

      sortLists(idx_entering_variable, idx_leaving_variable);   

      if(N[ N.size()-1 ] >= data->qtCols()) {
        N.conservativeResize(N.size()-1);
      } 

// */

// /*

      if(N[ idx_entering_variable ] >= data->qtCols()) {
        std::swap(N[ idx_entering_variable ], N[N.size()-1]);
        N.conservativeResize(N.size()-1);
        if(N.size() == data->qtCols() - data->qtRows()) data->resize();
      } 

      // sortLists(idx_entering_variable, idx_leaving_variable); 

// */
      // std::cout << "N: " << N.transpose() << "\n\nB: " << B.transpose() << "\n\n\n";

    } else {

      newEtaCol = false;
      
    }

  }

}


void Simplex::solve() {

  int n = data->qtCols();
  data->changeC(true);

  // std::cout << "n: " << n << std::endl;

  for(int i = 0; i < n; i++) {
    
    if(data->getUB(i) < INFTY) x[i] = data->getUB(i);
    else if(data->getLB(i) > -INFTY) x[i] = data->getLB(i);
    else x[i] = 0;

  }

  for(int i = 0; i < data->qtRows(); i++) {

    x[n+i] = data->multiplyByRow(x, i);

    if(x[n+i] >= 0) {

      data->setLB(n+i, 0);
      data->setUB(n+i, INFTY);
      data->setC(n+i, -1);

    } else {
      
      data->setLB(n+i, -INFTY);
      data->setUB(n+i, 0);
      data->setC(n+i, 1);

    }

  }

  Maximize();

  double auxObjValue = 0;
  for(int i = 0; i < data->qtRows(); i++) {
    if(std::abs(x[ i+n ]) < E2) continue;

    std::cout << "x[" << i+n << "]: " << x[i+n] << "; ";
    auxObjValue += ( x[ i+n ] * data->getC( i+n ) );

  } 

  std::cout << "auxObjValue: " << auxObjValue << std::endl;
  if(std::abs(auxObjValue) >= E1) {
    status = "Unfeasible";
    return;
  }

  data->changeC(false);

  std::cout << "PHASE TWO\nx: " << x.transpose() << std::endl;

  Maximize();

}


void Simplex::printSolution() {

  std::cout << "\nStatus: " << this->status << std::endl; 

  // if(status == "Optimal") {
    for(int i = 0; i < this->data->qtCols() - this->data->qtRows(); i++) {
      std::cout << "x_" << i+1 << ": " << x[i] << ";  ";
      value += x[i] * data->getC(i);
    }
    std::cout << "\n"; 
  // }

  std::cout << "\nObjective value: " << this->value << std::endl; 

}


inline double Simplex::getSolutionValue() {
  return this->value;
}
