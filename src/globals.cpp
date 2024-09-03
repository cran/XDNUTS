#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

//DEFINE THE GLOBAL VARIABLES AND HOW TO INTERACT WITH THEM

//initialize the divergent transitions matrix
arma::mat DT;

//initialize the number of divergent transitions to 0
unsigned int n_dt = 0;

//until we are in the sampling phase we say that we don't want to save them
//so we initialize store to false
bool store = false;
