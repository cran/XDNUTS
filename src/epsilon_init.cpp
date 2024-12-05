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
                    const unsigned int& k){
  
  //initialize the current value of the step size
  double eps = 1.0;
  
  //double or halve the stepsizes?
  double a = 1.0;
  
  //set the acceptance probability to zero
  double accept_prob = 0.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::sum(arma::square(m0.subvec(0,d-k-1)));
  
  //initialize the new energy level
  double H1;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step only for the continuous components
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += eps * m.subvec(0,d-k-1);
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::sum(arma::square(m.subvec(0,d-k-1)));
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(std::exp(H0-H1) > 0.5) - 1;
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  while( (count < 100) && (pow(accept_prob,a) > pow(0.5,a) || accept_prob == 0.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //set the acceptance probability to zero
    accept_prob = 0.0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += eps * m.subvec(0,d-k-1);
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //calculate the contribution to the sum of the metropolis log weights
    H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
      0.5 * arma::sum(arma::square(m.subvec(0,d-k-1)));
    
    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(H1)){
      H1 = arma::datum::inf;
      
      //if we were doubling, the last doubling was too much
      if(a == 1.0){
        //hence divide the step size and exit
        eps /= 2;
      }
      
      //exit
      break;
      
    }else{
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// diagonal matrix case for only the global stepsize
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::vec& M_inv_cont){
  
  //initialize the current value of the step size
  double eps = 1.0;
  
  //double or halve the stepsizes?
  double a = 1.0;
  
  //set the acceptance probability to zero
  double accept_prob = 0.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(arma::square( m0.subvec(0,d-k-1) ),M_inv_cont );
  
  //initialize the new energy level
  double H1;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step only for the continuous components
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += eps * M_inv_cont % m.subvec(0,d-k-1);
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(arma::square( m.subvec(0,d-k-1) ),M_inv_cont );
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(std::exp(H0-H1) > 0.5) - 1;
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  while( (count < 100) && (pow(accept_prob,a) > pow(0.5,a) || accept_prob == 0.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //set the acceptance probability to zero
    accept_prob = 0.0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += eps * M_inv_cont %  m.subvec(0,d-k-1);
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //calculate the contribution to the sum of the metropolis log weights
    H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
      0.5 * arma::dot(arma::square( m.subvec(0,d-k-1) ),M_inv_cont );
    
    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(H1)){
      H1 = arma::datum::inf;
      
      //if we were doubling, the last doubling was too much
      if(a == 1.0){
        //hence divide the step size and exit
        eps /= 2;
      }
      
      //exit
      break;
      
    }else{
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// dense matrix case for only the global stepsize
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    const arma::mat& M_inv_cont){
  
  //initialize the current value of the step size
  double eps = 1.0;
  
  //double or halve the stepsizes?
  double a = 1.0;
  
  //set the acceptance probability to zero
  double accept_prob = 0.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(m0.subvec(0,d-k-1), M_inv_cont * m0.subvec(0,d-k-1));
  
  //initialize the new energy level
  double H1;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step only for the continuous components
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += eps * M_inv_cont * m.subvec(0,d-k-1);
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(m.subvec(0,d-k-1), M_inv_cont * m.subvec(0,d-k-1));
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(std::exp(H0-H1) > 0.5) - 1;
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  while( (count < 100) && (pow(accept_prob,a) > pow(0.5,a) || accept_prob == 0.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //set the acceptance probability to zero
    accept_prob = 0.0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += eps * M_inv_cont *  m.subvec(0,d-k-1);
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //calculate the contribution to the sum of the metropolis log weights
    H1 =  Rcpp::as<double>(nlp(theta,args,true)) + 
      0.5 * arma::dot(m.subvec(0,d-k-1), M_inv_cont * m.subvec(0,d-k-1));
    
    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(H1)){
      H1 = arma::datum::inf;
      
      //if we were doubling, the last doubling was too much
      if(a == 1.0){
        //hence divide the step size and exit
        eps /= 2;
      }
      
      //exit
      break;
      
    }else{
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}


// ------------------------- DIFFERENT STEP SIZE ------------------------------

// identity matrix case for different stepsize
arma::vec init_epsilon(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc){
  
  //initialize the current value of the step sizes
  arma::vec eps = arma::ones<arma::vec>(k+1);
  
  //double or halve the stepsizes?
  arma::vec a = arma::ones<arma::vec>(k+1);
  
  //calculate the value of the potential energy
  double U = Rcpp::as<double>(nlp(theta0,args,true));
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U + 
    0.5 * arma::sum(arma::square(m0.subvec(0,d-k-1))) +
    arma::sum(arma::abs(m0.subvec(d-k,d-1)));
  
  //initialize the new energy level
  double H1;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step only for the continuous components
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += eps(0) * m.subvec(0,d-k-1);
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::sum(arma::square(m.subvec(0,d-k-1))) +
    arma::sum(arma::abs(m.subvec(d-k,d-1)));
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a(0) = 2.0 * static_cast<int>(std::exp(H0-H1) > 0.5) - 1;
  
  //reset the value of theta
  theta = theta0;
  
  //lets take a step for each discontinuous components
  unsigned int j;
  
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < k; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps(i+1) * segno(m(j));
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(!arma::is_finite(delta_U)){
      delta_U = arma::datum::inf;
    }
    
    //reset the discrete parameter
    theta(j) = theta_old;
    
    //if the probability of acceptance is greater than 0.5 then
    //double the stepsize, otherwise halve it
    a(1+i) = 2.0 * static_cast<int>(std::exp(-delta_U) > 0.5) - 1;
    
  }
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  //set the while conditions to one
  arma::vec conditions = arma::ones<arma::vec>(1+k);
  
  //initialize the current power
  double tmp_a;
  
  //initialize the current acceptance rate
  double tmp_acc;
  
  while( (count < 100) && arma::any(conditions == 1.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //check if the global condition has already been met
    if(conditions(0) == 1.0){
      
      //set the current power
      tmp_a = a(0);
      
      //initialize to zero the acceptance probability
      tmp_acc = 0.0;
      
      //double or halve the step size
      eps(0) *= pow(2,tmp_a);
      
      //let's take a leapfrog step
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //continuous parameter update by half step size
      theta.subvec(0,d-k-1) += eps(0) * m.subvec(0,d-k-1);
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //calculate the contribution to the sum of the metropolis log weights
      H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
        0.5 * arma::sum(arma::square(m.subvec(0,d-k-1))) +
        arma::sum(arma::abs(m.subvec(d-k,d-1)));
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!arma::is_finite(H1)){
        H1 = arma::datum::inf;
        
        //if we were doubling, the last doubling was too much
        if(tmp_a == 1.0){
          //hence divide the step size and exit
          eps(0) /= 2;
        }
        
        //exit
        conditions(0) = 0.0;
        
      }else{
        //set the acceptance probability
        //equal to the ratio between the exponential of the 2 energy levels
        tmp_acc = std::exp(H0-H1);
        conditions(0) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
      }
      
      //reset the value of theta
      theta = theta0;
      
    }
    
    for(unsigned int i = 0; i < k; i++){
      
      //check if the condition has been already met
      if(conditions(i+1) == 1.0){
        //set the current power
        tmp_a = a(i+1);
        
        //set the current acceptance probability
        tmp_acc = 0.0;
        
        //double or halve the step size
        eps(i+1) *= pow(2,tmp_a);
        
        //lets take a dhmc step:
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps(i+1) * segno(m(j));
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //check that it's finite
        if(!arma::is_finite(delta_U)){
          
          //make it inf
          delta_U = arma::datum::inf;
          
          //if we were doubling, the last doubling was too much
          if(tmp_a == 1.0){
            eps(i+1) /= 2;
          }
          
          //exit
          conditions(i+1) = 0.0;
          
        }else{
          //otherwise compute the acceptance probability and the condition to exit
          tmp_acc = std::exp(-delta_U);
          conditions(i+1) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
        }
        
        //reset the discrete parameter
        theta(j) = theta_old;
        
      }
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// diagonal matrix case for different stepsize
arma::vec init_epsilon(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv_cont,
                       const arma::vec& M_inv_disc){
  
  //initialize the current value of the step sizes
  arma::vec eps = arma::ones<arma::vec>(k+1);
  
  //double or halve the stepsizes?
  arma::vec a = arma::ones<arma::vec>(k+1);
  
  //calculate the value of the potential energy
  double U = Rcpp::as<double>(nlp(theta0,args,true));
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U + 
    0.5 * arma::dot(arma::square( m0.subvec(0,d-k-1) ),M_inv_cont ) + 
    arma::dot(arma::abs(m0.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the new energy level
  double H1;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step only for the continuous components
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += eps(0) * M_inv_cont % m.subvec(0,d-k-1);
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(arma::square( m.subvec(0,d-k-1) ),M_inv_cont ) + 
    arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a(0) = 2.0 * static_cast<int>(std::exp(H0-H1) > 0.5) - 1;
  
  //reset the value of theta
  theta = theta0;
  
  //lets take a step for each discontinuous components
  unsigned int j;
  
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < k; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps(i+1) * segno(m(j)) * M_inv_disc(j-d+k);
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(!arma::is_finite(delta_U)){
      delta_U = arma::datum::inf;
    }
    
    //reset the discrete parameter
    theta(j) = theta_old;
    
    //if the probability of acceptance is greater than 0.5 then
    //double the stepsize, otherwise halve it
    a(1+i) = 2.0 * static_cast<int>(std::exp(-delta_U) > 0.5) - 1;
    
  }
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  //set the while conditions to one
  arma::vec conditions = arma::ones<arma::vec>(1+k);
  
  //initialize the current power
  double tmp_a;
  
  //initialize the current acceptance rate
  double tmp_acc;
  
  while( (count < 100) && arma::any(conditions == 1.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //check if the global condition has already been met
    if(conditions(0) == 1.0){
      
      //set the current power
      tmp_a = a(0);
      
      //initialize to zero the acceptance probability
      tmp_acc = 0.0;
      
      //double or halve the step size
      eps(0) *= pow(2,tmp_a);
      
      //let's take a leapfrog step
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //continuous parameter update by half step size
      theta.subvec(0,d-k-1) += eps(0) * M_inv_cont % m.subvec(0,d-k-1);
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //calculate the contribution to the sum of the metropolis log weights
      H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
        0.5 * arma::dot(arma::square( m.subvec(0,d-k-1) ),M_inv_cont ) + 
        arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!arma::is_finite(H1)){
        H1 = arma::datum::inf;
        
        //if we were doubling, the last doubling was too much
        if(tmp_a == 1.0){
          //hence divide the step size and exit
          eps(0) /= 2;
        }
        
        //exit
        conditions(0) = 0.0;
        
      }else{
        //set the acceptance probability
        //equal to the ratio between the exponential of the 2 energy levels
        tmp_acc = std::exp(H0-H1);
        conditions(0) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
      }
      
      //reset the value of theta
      theta = theta0;
      
    }
    
    for(unsigned int i = 0; i < k; i++){
      
      //check if the condition has been already met
      if(conditions(i+1) == 1.0){
        //set the current power
        tmp_a = a(i+1);
        
        //set the current acceptance probability
        tmp_acc = 0.0;
        
        //double or halve the step size
        eps(i+1) *= pow(2,tmp_a);
        
        //lets take a dhmc step:
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps(i+1) * segno(m(j)) * M_inv_disc(j-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //check that it's finite
        if(!arma::is_finite(delta_U)){
          
          //make it inf
          delta_U = arma::datum::inf;
          
          //if we were doubling, the last doubling was too much
          if(tmp_a == 1.0){
            eps(i+1) /= 2;
          }
          
          //exit
          conditions(i+1) = 0.0;
          
        }else{
          //otherwise compute the acceptance probability and the condition to exit
          tmp_acc = std::exp(-delta_U);
          conditions(i+1) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
        }
        
        //reset the discrete parameter
        theta(j) = theta_old;
        
      }
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// dense matrix case for different stepsize
arma::vec init_epsilon(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const unsigned int& d,
                       const unsigned int& k,
                       arma::uvec& idx_disc,
                       const arma::mat& M_inv_cont,
                       const arma::vec& M_inv_disc){
  
  //initialize the current value of the step sizes
  arma::vec eps = arma::ones<arma::vec>(k+1);
  
  //double or halve the stepsizes?
  arma::vec a = arma::ones<arma::vec>(k+1);
  
  //calculate the value of the potential energy
  double U = Rcpp::as<double>(nlp(theta0,args,true));
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = U + 
    0.5 * arma::dot(m0.subvec(0,d-k-1), M_inv_cont * m0.subvec(0,d-k-1)) + 
    arma::dot(arma::abs(m0.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the new energy level
  double H1;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step only for the continuous components
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += eps(0) * M_inv_cont * m.subvec(0,d-k-1);
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(m.subvec(0,d-k-1), M_inv_cont * m.subvec(0,d-k-1)) +
    arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a(0) = 2.0 * static_cast<int>(std::exp(H0-H1) > 0.5) - 1;
  
  //reset the value of theta
  theta = theta0;
  
  //lets take a step for each discontinuous components
  unsigned int j;
  
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < k; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps(i+1) * segno(m(j)) * M_inv_disc(j-d+k);
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(!arma::is_finite(delta_U)){
      delta_U = arma::datum::inf;
    }
    
    //reset the discrete parameter
    theta(j) = theta_old;
    
    //if the probability of acceptance is greater than 0.5 then
    //double the stepsize, otherwise halve it
    a(1+i) = 2.0 * static_cast<int>(std::exp(-delta_U) > 0.5) - 1;
    
  }
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  //set the while conditions to one
  arma::vec conditions = arma::ones<arma::vec>(1+k);
  
  //initialize the current power
  double tmp_a;
  
  //initialize the current acceptance rate
  double tmp_acc;
  
  while( (count < 100) && arma::any(conditions == 1.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //check if the global condition has already been met
    if(conditions(0) == 1.0){
      
      //set the current power
      tmp_a = a(0);
      
      //initialize to zero the acceptance probability
      tmp_acc = 0.0;
      
      //double or halve the step size
      eps(0) *= pow(2,tmp_a);
      
      //let's take a leapfrog step
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //continuous parameter update by half step size
      theta.subvec(0,d-k-1) += eps(0) * M_inv_cont * m.subvec(0,d-k-1);
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps(0) * Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //calculate the contribution to the sum of the metropolis log weights
      H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
        0.5 * arma::dot(m.subvec(0,d-k-1), M_inv_cont * m.subvec(0,d-k-1)) +
        arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!arma::is_finite(H1)){
        H1 = arma::datum::inf;
        
        //if we were doubling, the last doubling was too much
        if(tmp_a == 1.0){
          //hence divide the step size and exit
          eps(0) /= 2;
        }
        
        //exit
        conditions(0) = 0.0;
        
      }else{
        //set the acceptance probability
        //equal to the ratio between the exponential of the 2 energy levels
        tmp_acc = std::exp(H0-H1);
        conditions(0) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
      }
      
      //reset the value of theta
      theta = theta0;
      
    }
    
    for(unsigned int i = 0; i < k; i++){
      
      //check if the condition has been already met
      if(conditions(i+1) == 1.0){
        //set the current power
        tmp_a = a(i+1);
        
        //set the current acceptance probability
        tmp_acc = 0.0;
        
        //double or halve the step size
        eps(i+1) *= pow(2,tmp_a);
        
        //lets take a dhmc step:
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps(i+1) * segno(m(j)) * M_inv_disc(j-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //check that it's finite
        if(!arma::is_finite(delta_U)){
          
          //make it inf
          delta_U = arma::datum::inf;
          
          //if we were doubling, the last doubling was too much
          if(tmp_a == 1.0){
            eps(i+1) /= 2;
          }
          
          //exit
          conditions(i+1) = 0.0;
          
        }else{
          //otherwise compute the acceptance probability and the condition to exit
          tmp_acc = std::exp(-delta_U);
          conditions(i+1) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
        }
        
        //reset the discrete parameter
        theta(j) = theta_old;
        
      }
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// ---------------------------------- K = D ------------------------------------

//identity
arma::vec init_epsilon(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const unsigned int& d,
                       arma::uvec& idx_disc){
  
  //initialize the current value of the step sizes
  arma::vec eps = arma::ones<arma::vec>(d);
  
  //double or halve the stepsizes?
  arma::vec a = arma::ones<arma::vec>(d);
  
  //calculate the value of the potential energy
  double U = Rcpp::as<double>(nlp(theta0,args,true));
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //lets take a step for each discontinuous components
  unsigned int j;
  
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < d; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps(i) * segno(m(j));
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(!arma::is_finite(delta_U)){
      delta_U = arma::datum::inf;
    }
    
    //reset the discrete parameter
    theta(j) = theta_old;
    
    //if the probability of acceptance is greater than 0.5 then
    //double the stepsize, otherwise halve it
    a(i) = 2.0 * static_cast<int>(std::exp(-delta_U) > 0.5) - 1;
    
  }
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  //set the while conditions to one
  arma::vec conditions = arma::ones<arma::vec>(d);
  
  //initialize the current power
  double tmp_a;
  
  //initialize the current acceptance rate
  double tmp_acc;
  
  while( (count < 100) && arma::any(conditions == 1.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    for(unsigned int i = 0; i < d; i++){
      
      //check if the condition has been already met
      if(conditions(i) == 1.0){
        //set the current power
        tmp_a = a(i);
        
        //set the current acceptance probability
        tmp_acc = 0.0;
        
        //double or halve the step size
        eps(i) *= pow(2,tmp_a);
        
        //lets take a dhmc step:
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps(i) * segno(m(j));
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //check that it's finite
        if(!arma::is_finite(delta_U)){
          
          //make it inf
          delta_U = arma::datum::inf;
          
          //if we were doubling, the last doubling was too much
          if(tmp_a == 1.0){
            eps(i) /= 2;
          }
          
          //exit
          conditions(i) = 0.0;
          
        }else{
          //otherwise compute the acceptance probability and the condition to exit
          tmp_acc = std::exp(-delta_U);
          conditions(i) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
        }
        
        //reset the discrete parameter
        theta(j) = theta_old;
        
      }
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
  
}

//diagonal
arma::vec init_epsilon(const arma::vec& theta0,
                       const arma::vec& m0,
                       const Rcpp::Function& nlp,
                       const Rcpp::List& args,
                       const unsigned int& d,
                       arma::uvec& idx_disc,
                       const arma::vec& M_inv){
  
  //initialize the current value of the step sizes
  arma::vec eps = arma::ones<arma::vec>(d);
  
  //double or halve the stepsizes?
  arma::vec a = arma::ones<arma::vec>(d);
  
  //calculate the value of the potential energy
  double U = Rcpp::as<double>(nlp(theta0,args,true));
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //lets take a step for each discontinuous components
  unsigned int j;
  
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < d; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps(i) * segno(m(j)) * M_inv(j);
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(!arma::is_finite(delta_U)){
      delta_U = arma::datum::inf;
    }
    
    //reset the discrete parameter
    theta(j) = theta_old;
    
    //if the probability of acceptance is greater than 0.5 then
    //double the stepsize, otherwise halve it
    a(i) = 2.0 * static_cast<int>(std::exp(-delta_U) > 0.5) - 1;
    
  }
  
  //continue in each directions until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  
  //set the while conditions to one
  arma::vec conditions = arma::ones<arma::vec>(d);
  
  //initialize the current power
  double tmp_a;
  
  //initialize the current acceptance rate
  double tmp_acc;
  
  while( (count < 100) && arma::any(conditions == 1.0)){
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    for(unsigned int i = 0; i < d; i++){
      
      //check if the condition has been already met
      if(conditions(i) == 1.0){
        //set the current power
        tmp_a = a(i);
        
        //set the current acceptance probability
        tmp_acc = 0.0;
        
        //double or halve the step size
        eps(i) *= pow(2,tmp_a);
        
        //lets take a dhmc step:
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps(i) * segno(m(j)) * M_inv(j);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //check that it's finite
        if(!arma::is_finite(delta_U)){
          
          //make it inf
          delta_U = arma::datum::inf;
          
          //if we were doubling, the last doubling was too much
          if(tmp_a == 1.0){
            eps(i) /= 2;
          }
          
          //exit
          conditions(i) = 0.0;
          
        }else{
          //otherwise compute the acceptance probability and the condition to exit
          tmp_acc = std::exp(-delta_U);
          conditions(i) = pow(tmp_acc,tmp_a) > pow(0.5,tmp_a) || tmp_acc == 0.0;
        }
        
        //reset the discrete parameter
        theta(j) = theta_old;
        
      }
    }
    
    //update the count
    count++;
  }
  
  //return a first estimate of the optimal step size
  return eps;
  
}
