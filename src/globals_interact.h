#ifndef GLOBALS_INTERACT_H
#define GLOBALS_INTERACT_H

#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// DEFINE THE FUNCTIONS THAT INTERACT WITH THE GLOBAL VARIABLES

//function that defines a global matrix of 10 rows and d columns
void create_DT(const unsigned int& d);

//function that enables archiving
void stora();

// function that returns the matrix of divergent transitions
arma::mat get_DT();

// function that returns the matrix of divergent transitions
unsigned int get_n_dt();

// function that adds a row to the divergent transitions matrix
void add_div_trans(const arma::subview_col<double>& x);

#endif // GLOBALS_INTERACT_H
