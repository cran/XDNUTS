#ifndef LEAPFROG_H
#define LEAPFROG_H

#include <iostream>
#include <RcppArmadillo.h>
#include <cmath>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "leapfrog.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS FOR MAKE A LEAPFROG STEP

//identity matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const unsigned int& k,
              arma::uvec idx_disc);

//diagonal matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const unsigned int& k,
              arma::uvec idx_disc,
              const arma::vec& M_inv_cont,
              const arma::vec& M_inv_disc);

//dense matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const unsigned int& k,
              arma::uvec idx_disc,
              const arma::mat& M_inv_cont,
              const arma::vec& M_inv_disc);

/* --------------------------------- CASO K = 0 ----------------------------- */

//identity matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d);

//diagonal matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const arma::vec& M_inv);

//dense matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const arma::mat& M_inv);

/* ------------------------------ CASO K = D -------------------------------- */

//identity matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              double& U,
              const unsigned int& d,
              arma::uvec& idx_disc);

//diagonal matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              double& U,
              const unsigned int& d,
              arma::uvec& idx_disc,
              const arma::vec& M_inv);

#endif // LEAPFROG_H
