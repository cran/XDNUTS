#ifndef EPSILON_INIT_H
#define EPSILON_INIT_H

#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "epsilon_init.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// FUNCTIONS THAT FIND A FIRST INITIAL VALUE FOR THE STEP SIZE

// ------------------------- ONLY GLOBAL STEP SIZE ---------------------------

// identity matrix case for only the global stepsize
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k);

// diagonal matrix case for only the global stepsize
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::vec& M_inv_cont);

// dense matrix case for only the global stepsize
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::mat& M_inv_cont);


// ------------------------- DIFFERENT STEP SIZE ------------------------------

// identity matrix case for different stepsize
arma::vec init_epsilon(const arma::vec& theta0,
                            const arma::vec& m0,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            const unsigned int& k,
                            arma::uvec& idx_disc);

// diagonal matrix case for different stepsize
arma::vec init_epsilon(const arma::vec& theta0,
                            const arma::vec& m0,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            const unsigned int& k,
                            arma::uvec& idx_disc,
                            const arma::vec& M_inv_cont,
                            const arma::vec& M_inv_disc);

// dense matrix case for different stepsize
arma::vec init_epsilon(const arma::vec& theta0,
                            const arma::vec& m0,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            const unsigned int& k,
                            arma::uvec& idx_disc,
                            const arma::mat& M_inv_cont,
                            const arma::vec& M_inv_disc);

// ---------------------------------- K = D ------------------------------------

//identity
arma::vec init_epsilon(const arma::vec& theta0,
                            const arma::vec& m0,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            arma::uvec& idx_disc);

//diagonal
arma::vec init_epsilon(const arma::vec& theta0,
                            const arma::vec& m0,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            arma::uvec& idx_disc,
                            const arma::vec& M_inv);


#endif // EPSILON_INIT_H
