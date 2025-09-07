#ifndef MCMC_H
#define MCMC_H

#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "recursive_tree.h"
#include "leapfrog.h"
#include "single_nuts.h"
#include "single_hmc.h"
#include "mcmc.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS THAT DO MCMC

// identity matrix case without recycling
void mcmc(arma::vec& theta0,
          const Rcpp::Function& nlp,
          const Rcpp::List& args,
          const unsigned int& N,
          const unsigned int& d,
          const unsigned int& k,
          arma::uvec& idx_disc,
          const unsigned int& max_treedepth,
          const double& refresh,
          arma::mat& out,
          const arma::vec& step_size,
          arma::uvec& step_length,
          arma::vec& energy,
          arma::vec& delta_energy,
          arma::vec& alpha,
          const bool warm_up,
          const unsigned int& chain_id,
          const int& thin,
          const double& log_tau,
          const unsigned int& L,
          const unsigned int& L_jitter);


/* -------------------------------------------------------------------------- */

// diagonal matrix case without recycling
void mcmc(arma::vec& theta0,
          const Rcpp::Function& nlp,
          const Rcpp::List& args,
          const unsigned int& N,
          const unsigned int& d,
          const unsigned int& k,
          arma::uvec& idx_disc,
          const unsigned int& max_treedepth,
          const double& refresh,
          arma::mat& out,
          const arma::vec& step_size,
          arma::uvec& step_length,
          arma::vec& energy,
          arma::vec& delta_energy,
          arma::vec& alpha,
          const bool warm_up,
          const unsigned int& chain_id,
          const int& thin,
          const double& log_tau,
          const unsigned int& L,
          const unsigned int& L_jitter,
          const arma::vec& M_cont,
          const arma::vec& M_disc,
          const arma::vec& M_inv_cont,
          const arma::vec& M_inv_disc);

/* -------------------------------------------------------------------------- */

// dense matrix case without recycling
void mcmc(arma::vec& theta0,
          const Rcpp::Function& nlp,
          const Rcpp::List& args,
          const unsigned int& N,
          const unsigned int& d,
          const unsigned int& k,
          arma::uvec& idx_disc,
          const unsigned int& max_treedepth,
          const double& refresh,
          arma::mat& out,
          const arma::vec& step_size,
          arma::uvec& step_length,
          arma::vec& energy,
          arma::vec& delta_energy,
          arma::vec& alpha,
          const bool warm_up,
          const unsigned int& chain_id,
          const int& thin,
          const double& log_tau,
          const unsigned int& L,
          const unsigned int& L_jitter,
          const arma::mat& M_cont,
          const arma::vec& M_disc,
          const arma::mat& M_inv_cont,
          const arma::vec& M_inv_disc);

/* ---------------------------- RECYCLED VERSION ---------------------------- */

// identity matrix case with recycling
void mcmc(arma::vec& theta0,
          const Rcpp::Function& nlp,
          const Rcpp::List& args,
          const unsigned int& N,
          const unsigned int& d,
          const unsigned int& k,
          arma::uvec& idx_disc,
          const unsigned int& max_treedepth,
          const double& refresh,
          arma::mat& out,
          const arma::vec& step_size,
          arma::uvec& step_length,
          arma::vec& energy,
          arma::vec& delta_energy,
          arma::vec& alpha,
          const bool warm_up,
          const unsigned int& chain_id,
          const double& log_tau,
          const unsigned int& L,
          const unsigned int& L_jitter,
          const unsigned int& K);

/* -------------------------------------------------------------------------- */

// diagonal matrix case with recycling
void mcmc(arma::vec& theta0,
          const Rcpp::Function& nlp,
          const Rcpp::List& args,
          const unsigned int& N,
          const unsigned int& d,
          const unsigned int& k,
          arma::uvec& idx_disc,
          const unsigned int& max_treedepth,
          const double& refresh,
          arma::mat& out,
          const arma::vec& step_size,
          arma::uvec& step_length,
          arma::vec& energy,
          arma::vec& delta_energy,
          arma::vec& alpha,
          const bool warm_up,
          const unsigned int& chain_id,
          const double& log_tau,
          const unsigned int& L,
          const unsigned int& L_jitter,
          const arma::vec& M_cont,
          const arma::vec& M_disc,
          const arma::vec& M_inv_cont,
          const arma::vec& M_inv_disc,
          const unsigned int& K);

/* -------------------------------------------------------------------------- */

// dense matrix case with recycling
void mcmc(arma::vec& theta0,
          const Rcpp::Function& nlp,
          const Rcpp::List& args,
          const unsigned int& N,
          const unsigned int& d,
          const unsigned int& k,
          arma::uvec& idx_disc,
          const unsigned int& max_treedepth,
          const double& refresh,
          arma::mat& out,
          const arma::vec& step_size,
          arma::uvec& step_length,
          arma::vec& energy,
          arma::vec& delta_energy,
          arma::vec& alpha,
          const bool warm_up,
          const unsigned int& chain_id,
          const double& log_tau,
          const unsigned int& L,
          const unsigned int& L_jitter,
          const arma::mat& M_cont,
          const arma::vec& M_disc,
          const arma::mat& M_inv_cont,
          const arma::vec& M_inv_disc,
          const unsigned int& K);

//wrapper function
void mcmc_wrapper(arma::mat& out,
                  arma::vec& step_size,
                  arma::uvec& step_length,
                  arma::vec& energy,
                  arma::vec& delta_energy,
                  arma::vec& alpha,
                  const unsigned int& max_treedepth,
                  const double& refresh,
                  arma::vec& theta,
                  const Rcpp::Function& nlp,
                  const Rcpp::List& args,
                  const unsigned int& N,
                  const double& bar,
                  const unsigned int& d,
                  const unsigned int& k,
                  arma::uvec& idx_disc,
                  arma::vec& M_cont_diag,
                  arma::vec& M_disc,
                  arma::vec& M_inv_cont_diag,
                  arma::vec& M_inv_disc,
                  arma::mat& M_cont_dense,
                  arma::mat& M_inv_cont_dense,
                  const unsigned int& K,
                  const std::string& M_type,
                  const bool warm_up,
                  const unsigned int& chain_id,
                  const int& thin,
                  const double& log_tau,
                  const unsigned int& L,
                  const unsigned int& L_jitter);

#endif // MCMC_H
