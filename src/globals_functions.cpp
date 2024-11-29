#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// FUNCTION THAT CALCULATES THE SIGN OF A SCALAR
int segno(const double& x){
  return (x > 0) - (x < 0);
}

// FUNCTION TO LOG SUM EXP WITH SIGN
void add_sign_log_sum_exp(double& log_abs_x,
                          double& sign_x,
                          const double& log_abs_y,
                          const double& sign_y){
  //save the maximum value between the two
  double max_val = std::max(log_abs_x,log_abs_y);
  
  //scale to the largest value and calculate the sum on the original scale
  double sum_exp_val = std::exp(log_abs_x - max_val) * sign_x + 
    std::exp(log_abs_y - max_val) * sign_y;
  
  //modify the values provided in input
  log_abs_x = max_val + std::log(std::abs(sum_exp_val));
  sign_x = segno(sum_exp_val);
}

// TERMINATION CRITERIA BASED ON THE INTERNAL PRODUCT

//identity matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const unsigned int& k){
  //if( dot(x.subvec(5*d,6*d-1),x.subvec(3*d,4*d-1)) < 0  || 
  //    dot(x.subvec(5*d,6*d-1),x.subvec(d,2*d-1)) > 0){
  if( 
    ( dot(x.subvec(5*d,6*d-k-1),x.subvec(3*d,4*d-k-1)) + 
      dot(x.subvec(6*d-k,6*d-1),arma::sign(x.subvec(4*d - k,4*d-1))) ) < 0 || 
      
    ( dot(x.subvec(5*d,6*d-k-1),x.subvec(d,2*d-k-1)) + 
      dot(x.subvec(6*d-k,6*d-1),arma::sign(x.subvec(2*d - k,2*d-1))) ) < 0
  ){  
    return 1.0;
  }
  
  return 0.0;
}

//diagonal matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::vec& M_inv_cont,
                    const arma::vec& M_inv_disc){
  //if( ( dot(M_inv_cont % x.subvec(5*d,6*d-k-1),x.subvec(3*d,4*d-k-1)) + dot(M_inv_disc % x.subvec(6*d-k,6*d-1),x.subvec(4*d-k,4*d-1)) ) < 0  || 
  //    ( dot(M_inv_cont % x.subvec(5*d,6*d-k-1),x.subvec(d,2*d-k-1))   + dot(M_inv_disc % x.subvec(6*d-k,6*d-1),x.subvec(2*d-k,2*d-1)) ) > 0){
  if( ( dot(M_inv_cont % x.subvec(5*d,6*d-k-1),x.subvec(3*d,4*d-k-1)) + 
      dot(M_inv_disc % x.subvec(6*d-k,6*d-1),    arma::sign(x.subvec(4*d-k,4*d-1))) ) < 0  || 
      ( dot(M_inv_cont % x.subvec(5*d,6*d-k-1),x.subvec(d,2*d-k-1))   + 
      dot(M_inv_disc % x.subvec(6*d-k,6*d-1),    arma::sign(x.subvec(2*d-k,2*d-1)) )) < 0){
  
    return 1.0;
  }
  
  return 0.0;
}

//dense matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::mat& M_inv_cont,
                    const arma::vec& M_inv_disc){

  //if( ( dot(M_inv_cont * x.subvec(5*d,6*d-k-1),x.subvec(3*d,4*d-k-1)) + dot(M_inv_disc % x.subvec(6*d-k,6*d-1),x.subvec(4*d-k,4*d-1)) ) < 0  || 
  //    ( dot(M_inv_cont * x.subvec(5*d,6*d-k-1),x.subvec(d,2*d-k-1))   + dot(M_inv_disc % x.subvec(6*d-k,6*d-1),x.subvec(2*d-k,2*d-1)) ) > 0){  
  if( ( dot(M_inv_cont * x.subvec(5*d,6*d-k-1),x.subvec(3*d,4*d-k-1)) + 
      dot(M_inv_disc % x.subvec(6*d-k,6*d-1),    arma::sign(x.subvec(4*d-k,4*d-1)) )) < 0  || 
        ( dot(M_inv_cont * x.subvec(5*d,6*d-k-1),x.subvec(d,2*d-k-1))   + 
        dot(M_inv_disc % x.subvec(6*d-k,6*d-1),    arma::sign(x.subvec(2*d-k,2*d-1)) )) < 0){  
      return 1.0;
  }
  
  return 0.0;
}

/* --------------------------- WITH RECYCLING ------------------------------- */

//identity matrix case with recycle
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& k,
                        const unsigned int& K){
  //if( dot(x.subvec((4+K)*d,(5+K)*d-1),x.subvec(3*d,4*d-1)) < 0  || 
  //    dot(x.subvec((4+K)*d,(5+K)*d-1),x.subvec(d,2*d-1)) > 0){
  if( ( dot(x.subvec((4+K)*d,(5+K)*d-k-1),x.subvec(3*d,4*d-k-1)) + 
      dot(x.subvec((5+K)*d-k,(5+K)*d-1),arma::sign(x.subvec(4*d-k,4*d-1))))< 0  || 
      
      (dot(x.subvec((4+K)*d,(5+K)*d-k-1),x.subvec(d,2*d-k-1)) + 
       dot(x.subvec((5+K)*d-k,(5+K)*d-1),arma::sign(x.subvec(2*d-k,2*d-1)))) < 0){  
    return 1.0;
  }
  
  return 0.0;
}

//diagonal matrix case with recycle
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& k,
                        const arma::vec& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K){
  if( ( dot(M_inv_cont % x.subvec((4+K)*d,(5+K)*d-k-1),x.subvec(3*d,4*d-k-1)) + 
        dot(M_inv_disc % x.subvec((5+K)*d-k,(5+K)*d-1),     arma::sign(x.subvec(4*d-k,4*d-1)) ) ) < 0  || 
      ( dot(M_inv_cont % x.subvec((4+K)*d,(5+K)*d-k-1),x.subvec(d,2*d-k-1))   + 
        dot(M_inv_disc % x.subvec((5+K)*d-k,(5+K)*d-1),     arma::sign(x.subvec(2*d-k,2*d-1)) )) < 0){

    return 1.0;
  }
  
  return 0.0;
}

//dense matrix case with recycle
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& k,
                        const arma::mat& M_inv_cont,
                        const arma::vec& M_inv_disc,
                        const unsigned int& K){
  
  if( ( dot(M_inv_cont * x.subvec((4+K)*d,(5+K)*d-k-1),x.subvec(3*d,4*d-k-1)) + 
        dot(M_inv_disc % x.subvec((5+K)*d-k,(5+K)*d-1),     arma::sign(x.subvec(4*d-k,4*d-1))) ) < 0  || 
      ( dot(M_inv_cont * x.subvec((4+K)*d,(5+K)*d-k-1),x.subvec(d,2*d-k-1))   + 
        dot(M_inv_disc % x.subvec((5+K)*d-k,(5+K)*d-1),     arma::sign(x.subvec(2*d-k,2*d-1)) )) < 0){  
    return 1.0;
  }
  
  return 0.0;
}

// Versions with k = 0 or d, classic NUTS or pure DNUTS

//identity matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d){
  if( dot(x.subvec(5*d,6*d-1),x.subvec(3*d,4*d-1)) < 0  || 
      dot(x.subvec(5*d,6*d-1),x.subvec(d,2*d-1)) < 0){
    
    return 1.0;
  }
  
  return 0.0;
}

//identity matrix case k = d uniform
double check_u_turn2(const arma::vec& x,
                     const unsigned int& d){
  if( dot(x.subvec(5*d,6*d-1),arma::sign(x.subvec(3*d,4*d-1))) < 0  || 
      dot(x.subvec(5*d,6*d-1),arma::sign(x.subvec(d,2*d-1))) < 0){
    
    return 1.0;
  }
  
  return 0.0;
}

//identity matrix, k = d, biased
double check_u_turn3(const arma::vec& x,
                     const unsigned int& d){
  if( dot(x.subvec(4*d,5*d-1),arma::sign(x.subvec(3*d,4*d-1))) < 0  || 
      dot(x.subvec(4*d,5*d-1),arma::sign(x.subvec(d,2*d-1))) < 0){
    
    return 1.0;
  }
  
  return 0.0;
}


// diagonal matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const arma::vec& M_inv){
  
  if( ( dot(M_inv % x.subvec(5*d,6*d-1),x.subvec(3*d,4*d-1)) ) < 0  || 
      ( dot(M_inv % x.subvec(5*d,6*d-1),x.subvec(d,2*d-1))  ) < 0){
    
    
    return 1.0;
  }
  
  return 0.0;
}

// diagonal matrix case k = d uniform
double check_u_turn2(const arma::vec& x,
                    const unsigned int& d,
                    const arma::vec& M_inv){
  
  if( ( dot(M_inv % x.subvec(5*d,6*d-1),     arma::sign(x.subvec(3*d,4*d-1)) )) < 0  || 
      ( dot(M_inv % x.subvec(5*d,6*d-1),     arma::sign(x.subvec(d,2*d-1))  )) < 0){
    
    
    return 1.0;
  }
  
  return 0.0;
}

// diagonal matrix case k = d biased
double check_u_turn3(const arma::vec& x,
                     const unsigned int& d,
                     const arma::vec& M_inv){
  
  if( ( dot(M_inv % x.subvec(4*d,5*d-1),     arma::sign(x.subvec(3*d,4*d-1)) )) < 0  || 
      ( dot(M_inv % x.subvec(4*d,5*d-1),     arma::sign(x.subvec(d,2*d-1))  )) < 0){
    
    
    return 1.0;
  }
  
  return 0.0;
}


// dense matrix case
double check_u_turn(const arma::vec& x,
                    const unsigned int& d,
                    const arma::mat& M_inv){
  
  if( ( dot(M_inv * x.subvec(5*d,6*d-1),x.subvec(3*d,4*d-1)) ) < 0  || 
      ( dot(M_inv * x.subvec(5*d,6*d-1),x.subvec(d,2*d-1))  ) < 0){

    return 1.0;
  }
  
  return 0.0;
}

/* --------------------------- WITH RECYCLING ------------------------------- */

// identity matrix case with recycling
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& K){
  if( dot(x.subvec((4+K)*d,(5+K)*d-1),x.subvec(3*d,4*d-1)) < 0  || 
      dot(x.subvec((4+K)*d,(5+K)*d-1),x.subvec(d,2*d-1)) < 0){
    
    return 1.0;
  }
  
  return 0.0;
}

// identity matrix case with recycling k = d
double check_u_turn_rec2(const arma::vec& x,
                        const unsigned int& d,
                        const unsigned int& K){
  if( dot(x.subvec((4+K)*d,(5+K)*d-1),arma::sign(x.subvec(3*d,4*d-1))) < 0  || 
      dot(x.subvec((4+K)*d,(5+K)*d-1),arma::sign(x.subvec(d,2*d-1))) < 0){
    
    return 1.0;
  }
  
  return 0.0;
}

// diagonal matrix case with recycling
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const arma::vec& M_inv,
                        const unsigned int& K){
  
  if( ( dot(M_inv % x.subvec((4+K)*d,(5+K)*d-1),x.subvec(3*d,4*d-1)) ) < 0  || 
      ( dot(M_inv % x.subvec((4+K)*d,(5+K)*d-1),x.subvec(d,2*d-1))  ) < 0){

    return 1.0;
  }
  
  return 0.0;
}

// diagonal matrix case with recycling k = d
double check_u_turn_rec2(const arma::vec& x,
                        const unsigned int& d,
                        const arma::vec& M_inv,
                        const unsigned int& K){
  
  if( ( dot(M_inv % x.subvec((4+K)*d,(5+K)*d-1),     arma::sign(x.subvec(3*d,4*d-1)) )) < 0  || 
      ( dot(M_inv % x.subvec((4+K)*d,(5+K)*d-1),     arma::sign(x.subvec(d,2*d-1))  ) ) < 0){
    
    return 1.0;
  }
  
  return 0.0;
}

// dense matrix case with recycling
double check_u_turn_rec(const arma::vec& x,
                        const unsigned int& d,
                        const arma::mat& M_inv,
                        const unsigned int& K){
  
  if( ( dot(M_inv * x.subvec((4+K)*d,(5+K)*d-1),x.subvec(3*d,4*d-1)) ) < 0  || 
      ( dot(M_inv * x.subvec((4+K)*d,(5+K)*d-1),x.subvec(d,2*d-1))  ) < 0){

    return 1.0;
  }
  
  return 0.0;
}

// FUNCTION TO SIMULATE FROM LAPLACE DISTRIBUTION
arma::vec rlaplace(const unsigned int& k){
  
  //simulate from a uniform in [-0.5,0.5]
  arma::vec u = arma::randu(k) - 0.5;
  
  //simulate from Laplace
  return -arma::sign(u) % arma::log(1-2*arma::abs(u));
}


// FUNCTION THAT CREATES A VECTOR OF INDEXES WITH WHICH TO RETURN A MESSAGE ON THE CONSOLE
arma::uvec sequenza(const unsigned int& N,const double& p){
  
  //divide N by p and take the smallest integer
  unsigned int n = std::floor(N * p);
  
  //check that it is admissible
  if(n == 0){
    arma::uvec x = {N+1};
    return x;
  }
  
  //calculate the length of the vector
  unsigned int K = std::floor(N / n);
  
  //check that it is admissible
  if(K == 0){
    arma::uvec x = {N+1};
    return x;
  }
  
  //create a vector of length K
  arma::uvec idx(K+1);
  
  //put the values inside
  for(unsigned int i = 0; i < K; i++){
    idx(i) = n*(i+1);
  }
  
  //return the sequence
  return idx;
}

// FUNCTION THAT INITIALIZES THE MASS MATRIX
void MM(arma::vec& M_cont_diag,
        arma::vec& M_disc,
        arma::vec& M_inv_cont_diag,
        arma::vec& M_inv_disc,
        arma::mat& M_cont_dense,
        arma::mat& M_inv_cont_dense,
        const Rcpp::List& control,
        const std::string& M_type){
  
  //let's compare all cases
  //if the matrix is specified we define the type of structure for the kinetic energy
  
  M_disc = arma::sqrt(Rcpp::as<arma::vec>(control["M_disc"]));
  M_inv_disc = 1.0 / M_disc;//Rcpp::as<arma::vec>(control["M_disc"]);
  
  //diagonal matrix
  if( M_type == "diagonal"){
    
    M_cont_diag = arma::sqrt(Rcpp::as<arma::vec>(control["M_cont"]));
    M_inv_cont_diag = 1.0 / Rcpp::as<arma::vec>(control["M_cont"]);
    
  }
  
  //dense matrix
  if( M_type == "dense"){
    
    M_cont_dense = arma::chol(Rcpp::as<arma::mat>(control["M_cont"])).t();
    M_inv_cont_dense = arma::inv(Rcpp::as<arma::mat>(control["M_cont"])); //+ 1e-6 * arma::eye<arma::mat>(d-k, d-k));
    
  }
  
}

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
               const std::string& M_type){
  
  //which iteration do we start from for estimation
  unsigned int init = N_adapt*K*bar;
  
  //compute the new warm up size
  unsigned int N_new = warm_up.n_rows-1-init;
  
  //throw an error if the sample are not enough for the estimate
  if( N_new < 2 ){
    Rcpp::stop("Insufficient warm-up samples to estimate the mass matrix! \nPlease consider decreasing the 'burn_adapt_ratio' or increasing 'N_adapt' in the 'control' argument using the 'set_parameter' function.");
  }
  
  //sample to use for the estimation
  arma::mat warm_up_subset = warm_up.rows(init,warm_up.n_rows-1);
  
  //covariance matrix of the posterior
  arma::mat M_inv = (N_new * arma::cov(warm_up_subset) + 5*1e-3*arma::eye<arma::mat>(d,d) )/ (N_new + 5);
  
  //diagonal of the posterior covariance matrix
  arma::vec diagonale = arma::diagvec(M_inv);
  
  //let's differentiate the various cases
  
  if(k == 0){
    // classic nuts
    //diagonal matrix
    if(M_type == "diagonal"){
      
      // continuous posterior variances
      M_inv_cont_diag = diagonale;
      
      //the inverse of the continuous posterior standard deviations
      M_cont_diag = 1.0 / arma::sqrt(diagonale);
    }
    
    //dense matrix
    if(M_type == "dense"){
      
      //posterior full covariance matrix of the continuous components
      M_inv_cont_dense = M_inv;
      
      //continuous mass matrix cholesky
      M_cont_dense = arma::chol(arma::inv(M_inv_cont_dense)).t();
      
    }
    
    
  }else if(k == d){
    // discontinuous posterior variances
    M_inv_disc = arma::sqrt(diagonale.subvec(0,d-1));
    
    //the inverse of the discontinuous posterior standard deviations
    M_disc = 1.0 / M_inv_disc;
    
  }else {
    // dnuts
    
    //discontinuous posterior variances
    M_inv_disc = arma::sqrt(diagonale.subvec(d-k,d-1));
    
    //the inverse of the discontinuous posterior standard deviations
    M_disc = 1.0 / M_inv_disc;
    
    //diagonal matrix
    if(M_type == "diagonal"){
      
      //continuous posterior variances
      M_inv_cont_diag = diagonale.subvec(0,d-k-1);
      
      //the inverse of the continuous posterior standard deviations
      M_cont_diag = 1.0 / arma::sqrt(diagonale.subvec(0,d-k-1));
    }
    
    //dense matrix
    if(M_type == "dense"){
      
      //posterior full covariance matrix of the continuous components
      M_inv_cont_dense = M_inv.submat(0,0,d-k-1,d-k-1);
      
      //continuous mass matrix cholesky
      M_cont_dense = arma::chol(arma::inv(M_inv_cont_dense)).t();
      
    }
  }
  
}

// FUNCTION THAT SAMPLES THE LENGTH OF THE TRAJECTORY IF WE ARE IN THE CLASSIC HMC
unsigned int sample_step_length(const unsigned int& L,
                                const unsigned int& L_jitter){
  //let's define the sampling extremes
  unsigned int lower;
  if(L <= L_jitter){
    lower = 1;
  }else{
    lower = L - L_jitter;
  }
  unsigned int upper = L + L_jitter;
  
  //generate a continuous uniform value in these extremes
  double u =  -0.5 + lower + arma::randu()*(1.0 + upper - lower);
  
  //round it up
  return std::round(u);
}
