#ifndef SINGLE_NUTS_H
#define SINGLE_NUTS_H

#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS FOR HAMILTONIAN MONTE CARLO WITH TERMINATION CRITERIA

/* --------------------------------- DNUTS ---------------------------------- */

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc);

/* -------------------------------------------------------------------------- */

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const arma::vec& M_inv_cont,
                        const arma::vec& M_inv_disc);

/* -------------------------------------------------------------------------- */

// dense matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const arma::mat& M_inv_cont,
                        const arma::vec& M_inv_disc);


/* ---------------------------- RECYCLED VERSION ---------------------------- */

// identity matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc, 
                        const unsigned int& K);

/* -------------------------------------------------------------------------- */

// diagonal matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const arma::vec& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K);


/* -------------------------------------------------------------------------- */

// dense matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const arma::mat& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* ---------------- VERSION WITH ONLY CONTINUOUS PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d);

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::vec& M_inv);

// dense matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::mat& M_inv);

/* ---------------------------- RECYCLED VERSION ---------------------------- */


// identity matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d, 
                        const unsigned int& K);

// diagonal matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::vec& M_inv,
                        const unsigned int& K);

// dense matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::mat& M_inv,
                        const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* --------------- VERSION WITH ONLY DISCONTINUOUS PARAMETERS --------------- */

/* -------------------------------------------------------------------------- */


// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc);

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc,
                        const arma::vec& M_inv);

/* ---------------------------- RECYCLED VERSION ---------------------------- */

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc,
                        const unsigned int& K);

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc,
                        const arma::vec& M_inv,
                        const unsigned int& K);


/* --------------------------------- XDHMC ---------------------------------- */


// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const double& log_tau);

/* -------------------------------------------------------------------------- */

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const double& log_tau,
                        const arma::vec& M_inv_cont,
                        const arma::vec& M_inv_disc);

/* -------------------------------------------------------------------------- */

// dense matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const double& log_tau,
                        const arma::mat& M_inv_cont,
                        const arma::vec& M_inv_disc);


/* ---------------------------- RECYCLED VERSION ---------------------------- */


// identity matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const double& log_tau, 
                        const unsigned int& K);

/* -------------------------------------------------------------------------- */

// diagonal matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const double& log_tau,
                        const arma::vec& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K);


/* -------------------------------------------------------------------------- */

// dense matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc,
                        const double& log_tau,
                        const arma::mat& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* ---------------- VERSION WITH ONLY CONTINUOUS PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau);

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau,
                        const arma::vec& M_inv);

// dense matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau,
                        const arma::mat& M_inv);

/* ---------------------------- RECYCLED VERSION ---------------------------- */


// identity matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau, 
                        const unsigned int& K);

// diagonal matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau,
                        const arma::vec& M_inv,
                        const unsigned int& K);

// dense matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau,
                        const arma::mat& M_inv,
                        const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* --------------- VERSION WITH ONLY DISCONTINUOUS PARAMETERS --------------- */

/* -------------------------------------------------------------------------- */


// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc,
                        const double& tau);

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc,
                        const double& tau,
                        const arma::vec& M_inv);

/* ---------------------------- RECYCLED VERSION ---------------------------- */

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc,
                        const double& tau,
                        const unsigned int& K);

// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        arma::uvec& idx_disc, 
                        const double& tau,
                        const arma::vec& M_inv,
                        const unsigned int& K);



#endif // SINGLE_NUTS_H
