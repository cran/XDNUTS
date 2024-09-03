#ifndef GLOBALS_H
#define GLOBALS_H

#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

//DEFINE THE GLOBAL VARIABLES AND HOW TO INTERACT WITH THEM

//export divergent transitions matrix
extern arma::mat DT;

//export the variable that counts the number of divergent transitions during
//sampling phase
extern unsigned int n_dt;

//global variable that decides whether to save divergent transitions or not
// (in the warm up phase it makes no sense to save them)
extern bool store;

#endif // GLOBALS_H
