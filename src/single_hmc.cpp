#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "leapfrog.h"
#include "single_hmc.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// FUNCTIONS FOR CLASSIC HAMILTONIAN MONTE CARLO

// identity matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& L,
                        const unsigned int& d,
                        const unsigned int& k,
                        arma::uvec& idx_disc){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::sum(arma::square(m_m.subvec(0,d-k-1))) + 
    arma::sum(arma::abs(m_m.subvec(d-k,d-1)));
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 

  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(k+1);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,k,idx_disc);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_p;
      }
      
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,k,idx_disc);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_m;
      }
    }
    
    //update the log sum of the multinomial weights
    lsw = arma::log_add_exp(lsw,lsw2);
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}


/* -------------------------------------------------------------------------- */

// diagonal matrix case
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
                       const arma::vec& M_inv_disc){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(arma::square(m_m.subvec(0,d-k-1)),M_inv_cont) + 
    arma::dot(arma::abs(m_m.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(k+1);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_p;
      }
      
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_m;
      }
    }
    
    //update the log sum of the multinomial weights
    lsw = arma::log_add_exp(lsw,lsw2);
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

/* -------------------------------------------------------------------------- */

// dense matrix case
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
                       const arma::vec& M_inv_disc){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(m_m.subvec(0,d-k-1),M_inv_cont * m_m.subvec(0,d-k-1)) + 
    arma::dot(arma::abs(m_m.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(k+1);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_p;
      }
      
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_m;
      }
    }
    
    //update the log sum of the multinomial weights
    lsw = arma::log_add_exp(lsw,lsw2);
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}


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
                       const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::sum(arma::square(m_m.subvec(0,d-k-1))) + 
    arma::sum(arma::abs(m_m.subvec(d-k,d-1)));
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(k+1);
  
  double alpha_tmp = 0.0;
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,k,idx_disc);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_p;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,k,idx_disc);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_m;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
      
    }
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}


/* -------------------------------------------------------------------------- */

// diagonal matrix case with recycle
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
                        const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(arma::square(m_m.subvec(0,d-k-1)),M_inv_cont) + 
    arma::dot(arma::abs(m_m.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(k+1);
  
  double alpha_tmp = 0.0;
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_p;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_m;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
      
    }
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}


/* -------------------------------------------------------------------------- */

// dense matrix case with recycle
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
                        const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(m_m.subvec(0,d-k-1),M_inv_cont * m_m.subvec(0,d-k-1)) + 
    arma::dot(arma::abs(m_m.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(k+1);
  
  double alpha_tmp = 0.0;
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_p;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_m;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
      
    }

    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

/* -------------------------------------------------------------------------- */

/* --------------------------- CASE WITH k = 0 ------------------------------ */

/* -------------------------------------------------------------------------- */

// identity matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& L,
                        const unsigned int& d){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::sum(arma::square(m_m));
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(1);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_p;
      }
      
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_m;
      }
    }
    
    //update the log sum of the multinomial weights
    lsw = arma::log_add_exp(lsw,lsw2);
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::vec& M_inv){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(arma::square(m_m),M_inv);
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(1);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,M_inv);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_p;
      }
      
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,M_inv);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_m;
      }
    }
    
    //update the log sum of the multinomial weights
    lsw = arma::log_add_exp(lsw,lsw2);
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

// dense matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::mat& M_inv){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(m_m,M_inv * m_m);
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(1);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,M_inv);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_p;
      }
      
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,M_inv);
      
      //update the proposed value
      if(arma::randu() < std::exp(lsw2 - lsw)){
        theta = theta_m;
      }
    }
    
    //update the log sum of the multinomial weights
    lsw = arma::log_add_exp(lsw,lsw2);
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

/* ---------------------------- RECYCLED VERSION ---------------------------- */


// identity matrix case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::sum(arma::square(m_m));
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(1);
  
  double alpha_tmp = 0.0;
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_p;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_m;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
      
    }
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::vec& M_inv,
                       const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(arma::square(m_m),M_inv);
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(1);
  
  double alpha_tmp = 0.0;
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,M_inv);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_p;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,M_inv);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_m;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
      
    }
    
    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

// dense matrix case with recycle
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       const arma::mat& M_inv,
                       const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(m_m,M_inv * m_m);
  
  //initialize the sum of the logarithm weight for multinomial sampling from the trajectory
  double lsw = -H0;
  double lsw2 = -arma::datum::inf; 
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance rates
  arma::vec alpha = arma::zeros<arma::vec>(1);
  
  double alpha_tmp = 0.0;
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,lsw2,alpha,eps,nlp,args,H0,d,M_inv);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_p;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,lsw2,alpha,-eps,nlp,args,H0,d,M_inv);
      
      //calculate the probability of acceptance metropolis
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //update the proposed values
      
      //biased for the first
      if(arma::randu() < alpha_tmp){
        theta.subvec(0,d-1) = theta_m;
      }
      
      //update the log sum of the multinomial weights
      lsw = arma::log_add_exp(lsw,lsw2);
      alpha_tmp = std::exp(lsw2 - lsw);
      
      //uniform for the others
      for(unsigned int i = 1; i < K; i++){
        if(arma::randu() < alpha_tmp){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
      
    }

    //increase trajectory length
    step_length++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

/* -------------------------------------------------------------------------- */

/* ------------------ VERSION WITH ONLY DISCRETE PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */


// identity matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the value of the current potential energy
  double U_m = Rcpp::as<double>(nlp(theta,args,true));
  double U_p = U_m;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U_m + arma::sum(arma::abs(m_m));
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance ratess
  arma::vec alpha = arma::zeros<arma::vec>(d);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      //leapfrog(theta_p,m_p,alpha,eps,nlp,args,U,d,idx_disc);
      leapfrog(theta_p,m_p,alpha,eps,nlp,args,U_p,d,idx_disc);
      
      //biased progressive sampling
      if(arma::randu() < 1.0/(1+step_length)){
          theta = theta_p;
      }
      
    }else{
      //continue the trajectory backward
      //leapfrog(theta_m,m_m,alpha,-eps,nlp,args,U,d,idx_disc);
      leapfrog(theta_m,m_m,alpha,-eps,nlp,args,U_m,d,idx_disc);
      
      //biased progressive sampling
      if(arma::randu() < 1.0/(1+step_length)){
        theta = theta_m;
      }
    }
    
    //increase trajectory length
    step_length++;
  }
  
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the value proposed by the trajectory
  arma::vec theta = theta0;
  
  //calculate the value of the current potential energy
  double U_m = Rcpp::as<double>(nlp(theta,args,true));
  double U_p = U_m;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U_m + arma::sum(arma::abs(m_m));
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance ratess
  arma::vec alpha = arma::zeros<arma::vec>(d);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,alpha,eps,nlp,args,U_p,d,idx_disc,M_inv);
      
      //biased progressive sampling
      if(arma::randu() < 1.0/(1+step_length)){
        theta = theta_p;
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,alpha,-eps,nlp,args,U_m,d,idx_disc,M_inv);
      
      //biased progressive sampling
      if(arma::randu() < 1.0/(1+step_length)){
        theta = theta_m;
      }
    }
    
    //increase trajectory length
    step_length++;
  }
  
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}


/* ---------------------------- RECYCLED VERSION ---------------------------- */

// identity matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the value of the current potential energy
  double U_m = Rcpp::as<double>(nlp(theta0,args,true));
  double U_p = U_m;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U_m + arma::sum(arma::abs(m_m));
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance ratess
  arma::vec alpha = arma::zeros<arma::vec>(d);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,alpha,eps,nlp,args,U_p,d,idx_disc);
      
      //update the proposed values
      //biased sampling for all
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < 1.0/(1+step_length)){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,alpha,-eps,nlp,args,U_m,d,idx_disc);
      
      //update the proposed values
      //biased sampling for all
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < 1.0/(1+step_length)){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
    }
    
    //increase trajectory length
    step_length++;
  }
  
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List hmc_singolo(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const double& eps,
                       const unsigned int& L,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv,
                       const unsigned int& K){
  
  //initialization of the extreme values of the trajectory
  arma::vec theta_m = theta0;
  arma::vec m_m = m0;
  arma::vec theta_p = theta0;
  arma::vec m_p = m0;
  
  //initialization of the values proposed by the trajectory
  arma::vec theta(d*K);
  for(unsigned int i = 0; i < K; i++){
    theta.subvec(i*d,(i+1)*d - 1) = theta0;
  }
  
  //calculate the value of the current potential energy
  double U_m = Rcpp::as<double>(nlp(theta0,args,true));
  double U_p = U_m;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U_m + arma::sum(arma::abs(m_m));
  
  //length of the trajectory 
  unsigned int step_length = 0;
  
  //metropolis acceptance ratess
  arma::vec alpha = arma::zeros<arma::vec>(d);
  
  //create the trajectory
  while(step_length < L){
    
    //should the doubling be done on the right or left?
    if(arma::randu() > 0.5){
      //continue the trajectory forward
      leapfrog(theta_p,m_p,alpha,eps,nlp,args,U_p,d,idx_disc,M_inv);
      
      //update the proposed values
      //biased sampling for all
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < 1.0/(1+step_length)){
          theta.subvec(i*d,(i+1)*d-1) = theta_p;
        }
      }
      
    }else{
      //continue the trajectory backward
      leapfrog(theta_m,m_m,alpha,-eps,nlp,args,U_m,d,idx_disc,M_inv);
      
      //update the proposed values
      //biased sampling for all
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < 1.0/(1+step_length)){
          theta.subvec(i*d,(i+1)*d-1) = theta_m;
        }
      }
    }
    
    //increase trajectory length
    step_length++;
  }
  
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("alpha") = alpha / step_length,
                            Rcpp::Named("n") = step_length,
                            Rcpp::Named("E") = H0);
}
