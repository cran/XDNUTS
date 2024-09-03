#ifndef EPSILON_INIT_H
#define EPSILON_INIT_H

#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS THAT FIND A FIRST INITIAL VALUE FOR THE STEP SIZE

// identity matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc);

/* -------------------------------------------------------------------------- */

// diagonal matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const arma::vec& M_inv_cont,
                    const arma::vec& M_inv_disc);


/* -------------------------------------------------------------------------- */

// dense matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const arma::mat& M_inv_cont,
                    const arma::vec& M_inv_disc);

/* -------------------------------------------------------------------------- */

/* ------------------------ VERSION WITH k = 0 ------------------------------ */

/* -------------------------------------------------------------------------- */

// identity matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d);

// diagonal matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const arma::vec& M_inv);

// dense matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const arma::mat& M_inv);

/* -------------------------------------------------------------------------- */

/* ------------------------ VERSION WITH k = d ------------------------------ */

/* -------------------------------------------------------------------------- */

// identity matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    arma::uvec& idx_disc);

// diagonal matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    arma::uvec& idx_disc,
                    const arma::vec& M_inv);

#endif // EPSILON_INIT_H
