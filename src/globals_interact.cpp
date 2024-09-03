#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// DEFINE THE FUNCTIONS THAT INTERACT WITH THE GLOBAL VARIABLES

//function that defines a global matrix of 10 rows and d columns
void create_DT(const unsigned int& d){
  DT = arma::zeros<arma::mat>(10,d);
  store = false;
  n_dt = 0;
}

//function that enables archiving
void stora(){
  store = true;
}

// function that returns the matrix of divergent transitions
arma::mat get_DT(){
  return DT.rows(0,n_dt-1);
}

// function that adds a row to the divergent transitions matrix
void add_div_trans(const arma::subview_col<double>& x){
  if(store){
    if(n_dt >= DT.n_rows){
      DT.resize(DT.n_rows + 10,DT.n_cols);
    }
    DT.row(n_dt) = x.t();
    n_dt++;
  }
}
