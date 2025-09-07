#ifndef SINGLE_HMC_H
#define SINGLE_HMC_H

#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "leapfrog.h"
#include "single_hmc.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS FOR CLASSIC HAMILTONIAN MONTE CARLO

// identity matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc);


/* -------------------------------------------------------------------------- */

// diagonal matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv_cont,
                       const arma::vec& M_inv_disc);

/* -------------------------------------------------------------------------- */

// dense matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const arma::mat& M_inv_cont,
                       const arma::vec& M_inv_disc);


/* ---------------------------- RECYCLED VERSION ---------------------------- */

//identity matrix case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const unsigned int& K);


/* -------------------------------------------------------------------------- */

// diagonal matrx case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv_cont,
                       const arma::vec& M_inv_disc,
                       const unsigned int& K);


/* -------------------------------------------------------------------------- */

// dense matrx case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const arma::mat& M_inv_cont,
                       const arma::vec& M_inv_disc,
                       const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* --------------------------- CASE WITH k = 0 ------------------------------ */

/* -------------------------------------------------------------------------- */


// identity matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d);

// diagonal matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::vec& M_inv);

// dense matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::mat& M_inv);

/* ---------------------------- RECYCLED VERSION ---------------------------- */


// identity matrx case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& K);

// diagonal matrx case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::vec& M_inv,
                       const unsigned int& K);

// dense matrx case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::mat& M_inv,
                       const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* ------------------ VERSION WITH ONLY DISCRETE PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */

// identity matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc);

// diagonal matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv);


/* ---------------------------- RECYCLED VERSION ---------------------------- */

// identity matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const unsigned int& K);

// identity matrx case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv,
                       const unsigned int& K);

#endif // SINGLE_HMC_H
