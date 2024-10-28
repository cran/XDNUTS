#ifndef GLOBALS_FUNCTIONS_H
#define GLOBALS_FUNCTIONS_H

#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTION THAT CALCULATES THE SIGN OF A SCALAR
int segno(const double& x);

// FUNCTION TO LOG SUM EXP WITH SIGN
void add_sign_log_sum_exp(double& log_abs_x,
                          double& sign_x,
                          const double& log_abs_y,
                          const double& sign_y);

// TERMINATION CRITERIA BASED ON THE INTERNAL PRODUCT

//identity matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const unsigned int& k);

//diagonal matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::vec& M_inv_cont,
                    const arma::vec& M_inv_disc);

//dense matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::mat& M_inv_cont,
                    const arma::vec& M_inv_disc);

/* --------------------------- WITH RECYCLING ------------------------------- */

//identity matrix case with recycle
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& k,
                        const unsigned int& K);

//diagonal matrix case with recycle
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& k,
                        const arma::vec& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K);

//dense matrix case with recycle
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& k,
                        const arma::mat& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K);

// Versions with k = 0 or d, classic NUTS or pure DNUTS

//identity matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d);

//identity matrix case k = d, uniform
double check_u_turn2(const arma::vec& x,
                     const unsigned int& d);

//identity matrix case k = d, biased
double check_u_turn3(const arma::vec& x,
                     const unsigned int& d);

// diagonal matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const arma::vec& M_inv);

// diagonal matrix case k = d, uniform
double check_u_turn2(const arma::vec& x,
                     const unsigned int& d,
                     const arma::vec& M_inv);

//identity matrix case k = d, biased
double check_u_turn3(const arma::vec& x,
                     const unsigned int& d,
                     const arma::vec& M_inv);

// dense matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const arma::mat& M_inv);

/* --------------------------- WITH RECYCLING ------------------------------- */

// identity matrix case with recycling
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& K);

// identity matrix case with recycling k = d
double check_u_turn_rec2(const arma::vec& x,
                         const unsigned int& d,
                         const unsigned int& K);

// diagonal matrix case with recycling
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const arma::vec& M_inv,
                        const unsigned int& K);

// diagonal matrix case with recycling k = d
double check_u_turn_rec2(const arma::vec& x,
                         const unsigned int& d,
                         const arma::vec& M_inv,
                         const unsigned int& K);

// dense matrix case with recycling
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const arma::mat& M_inv,
                        const unsigned int& K);

// FUNCTION TO SIMULATE FROM LAPLACE DISTRIBUTION
arma::vec rlaplace(const unsigned int& k);


// FUNCTION THAT CREATES A VECTOR OF INDEXES WITH WHICH TO RETURN A MESSAGE ON THE CONSOLE
arma::uvec sequenza(const unsigned int& N,const double& p);

// FUNCTION THAT INITIALIZES THE MASS MATRIX
void MM(arma::vec& M_cont_diag,
        arma::vec& M_disc,
        arma::vec& M_inv_cont_diag,
        arma::vec& M_inv_disc,
        arma::mat& M_cont_dense,
        arma::mat& M_inv_cont_dense,
        const Rcpp::List& control,
        const std::string& M_type);

// FUNCTION THAT UPDATE THE ESTIMATE OF THE MASS MATRIX AFTER THE WARM UP PHASE
void update_MM(arma::vec& M_cont_diag,
               arma::vec& M_disc,
               arma::vec& M_inv_cont_diag,
               arma::vec& M_inv_disc,
               arma::mat& M_cont_dense,
               arma::mat& M_inv_cont_dense,
               const arma::mat& warm_up,
               const unsigned int& N_adapt,
               const unsigned int& K,
               const double bar,
               const unsigned int& d,
               const unsigned int& k,
               const std::string& M_type);

// FUNCTION THAT SAMPLES THE LENGTH OF THE TRAJECTORY IF WE ARE IN THE CLASSIC HMC
unsigned int sample_step_length(const unsigned int& L,
                                const unsigned int& L_jitter);

#endif // GLOBALS_FUNCTIONS_H
