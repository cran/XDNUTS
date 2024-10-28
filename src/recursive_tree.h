#ifndef RECURSIVE_TREE_H
#define RECURSIVE_TREE_H

#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS TO BUILD THE TRAJECTORY PROGRESSIVELY

/* ----------------------------------- DNUTS -------------------------------- */

// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc);

/*----------------------------------------------------------------------------*/


// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const arma::vec& M_inv_cont,
                     const arma::vec& M_inv_disc);

/*----------------------------------------------------------------------------*/


// dense matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const arma::mat& M_inv_cont,
                     const arma::vec& M_inv_disc);


/* --------------------------   RECYCLED VERSION   -------------------------- */

// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const unsigned int& K);


/*----------------------------------------------------------------------------*/


// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const arma::vec& M_inv_cont,
                     const arma::vec& M_inv_disc,
                     const unsigned int& K);

/*----------------------------------------------------------------------------*/


// dense matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const arma::mat& M_inv_cont,
                     const arma::vec& M_inv_disc,
                     const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* ---------------- VERSION WITH ONLY CONTINUOUS PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */

// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d);

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::vec& M_inv);

// dense matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::mat& M_inv);


/* --------------------------   RECYCLED VERSION   -------------------------- */

// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& K);

/*----------------------------------------------------------------------------*/


// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::vec& M_inv,
                     const unsigned int& K);

// dense matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::mat& M_inv,
                     const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* ------------------ VERSION WITH ONLY DISCRETE PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */

// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc);

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const arma::vec& M_inv);

/* -------------------------- RECYCLED VERSION ----------------------------- */


// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const unsigned int& K);

// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const arma::vec& M_inv,
                     const unsigned int& K);

/* ---------------------------------- XDHMC --------------------------------- */

// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const double& log_tau);

/*----------------------------------------------------------------------------*/


// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const arma::vec& M_inv_cont,
                     const arma::vec& M_inv_disc);

/*----------------------------------------------------------------------------*/


// dense matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const arma::mat& M_inv_cont,
                     const arma::vec& M_inv_disc);


/* --------------------------   RECYCLED VERSION   -------------------------- */

// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const unsigned int& K);


/*----------------------------------------------------------------------------*/


// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const arma::vec& M_inv_cont,
                     const arma::vec& M_inv_disc,
                     const unsigned int& K);

/*----------------------------------------------------------------------------*/


// dense matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
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

// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau);

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const arma::vec& M_inv);

// dense matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const arma::mat& M_inv);


/* --------------------------   RECYCLED VERSION   -------------------------- */

// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const unsigned int& K);

/*----------------------------------------------------------------------------*/


// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const arma::vec& M_inv,
                     const unsigned int& K);

// dense matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const arma::mat& M_inv,
                     const unsigned int& K);

/* -------------------------------------------------------------------------- */

/* ------------------ VERSION WITH ONLY DISCRETE PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */


// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& tau);

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& tau,
                     const arma::vec& M_inv);

/* -------------------------- RECYCLED VERSION ----------------------------- */


// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& tau,
                     const unsigned int& K);

// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& tau,
                     const arma::vec& M_inv,
                     const unsigned int& K);


#endif // RECURSIVE_TREE_H
