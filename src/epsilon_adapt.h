#ifndef EPSILON_ADAPT_H
#define EPSILON_ADAPT_H

#include <iostream>
#include <RcppArmadillo.h>
#include <cmath>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "recursive_tree.h"
#include "leapfrog.h"
#include "single_hmc.h"
#include "single_nuts.h"
#include "epsilon_init.h"
#include "epsilon_adapt.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTION FOR STEP SIZE CALIBRATION

// identity matrix case
void adapt_stepsize(arma::vec& theta0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    double& eps0,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const Rcpp::List& control,
                    const unsigned int& N,
                    const double& log_tau,
                    const unsigned int& L);

/* -------------------------------------------------------------------------- */


// diagonal matrix case
void adapt_stepsize(arma::vec& theta0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    double& eps0,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const Rcpp::List& control,
                    const unsigned int& N,
                    const double& log_tau,
                    const unsigned int& L,
                    const arma::vec& M_cont,
                    arma::vec& M_disc,
                    const arma::vec& M_inv_cont,
                    arma::vec& M_inv_disc);

/* -------------------------------------------------------------------------- */


// dense matrix case
void adapt_stepsize(arma::vec& theta0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    double& eps0,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const Rcpp::List& control,
                    const unsigned int& N,
                    const double& log_tau,
                    const unsigned int& L,
                    const arma::mat& M_cont,
                    arma::vec& M_disc,
                    const arma::mat& M_inv_cont,
                    arma::vec& M_inv_disc);

// WRAPPER FUNCTION

void adapt_stepsize_wrapper(arma::vec& theta,
                            double& eps,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            const unsigned int& k,
                            arma::uvec& idx_disc,
                            const unsigned int& N_init,
                            const Rcpp::List& control,
                            const arma::vec& M_cont_diag,
                            arma::vec& M_disc,
                            const arma::vec& M_inv_cont_diag,
                            arma::vec& M_inv_disc,
                            const arma::mat& M_cont_dense,
                            const arma::mat& M_inv_cont_dense,
                            const std::string& M_type,
                            const double& log_tau,
                            const unsigned int& L,
                            const bool& verbose,
                            const unsigned int& chain_id);


#endif // EPSILON_ADAPT_H
