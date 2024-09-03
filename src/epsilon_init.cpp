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

// identity matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::sum(arma::square(m0.subvec(0,d-k-1))) +
    arma::sum(arma::abs(m0.subvec(d-k,d-1)));
  
  //initialize the new energy level
  double H1;
  
  //initialize the potential energy
  double U;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += 0.5 * eps * m.subvec(0,d-k-1);
  
  //calculate the value of the new potential energy
  U = Rcpp::as<double>(nlp(theta,args,true));
  
  // if the potential energy is finite then we can continue
  if(arma::is_finite(U)){
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    unsigned int j;
    
    //loop for each discontinuous coordinate
    for(unsigned int i = 0; i < k; i++){
      
      //set the index
      j = idx_disc(i);
      
      //modify the discrete parameter
      theta_old = theta(j);
      
      theta(j) = theta_old + eps * segno(m(j));
      
      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
      
      //refraction or reflection?
      if( std::abs(m(j)) > delta_U ){
        
        //refraction
        m(j) -= segno(m(j)) * delta_U;
        U += delta_U;
        
      }else{
        
        //reflection
        theta(j) = theta_old;
        m(j) *= -1.0;
        
      }
      
    }
    
    // continue updating continuous parameters
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * m.subvec(0,d-k-1);
    
    //calculate the gradient
    arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //let's make sure it's finished
    if(arma::is_finite(grad)){
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps * grad;
      
      //calculate the contribution to the sum of the metropolis log weights
      H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
        0.5 * arma::sum(arma::square(m.subvec(0,d-k-1))) +
        arma::sum(arma::abs(m.subvec(d-k,d-1)));
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!arma::is_finite(H1)){
        H1 = arma::datum::inf;
      }
      
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);
      
      //if the probability of acceptance is greater than 0.5 then
      //double the stepsize, otherwise halve it
      a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
    }
    
  }
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;
  
    //step forward with a modified step size
    accept_prob = 0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * m.subvec(0,d-k-1);
    
    //calculate the value of the new potential energy
    U = Rcpp::as<double>(nlp(theta,args,true));
    
    // if the potential energy is finite then we can continue
    if(arma::is_finite(U)){
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      
      //loop for each discontinuous coordinate
      for(unsigned int i = 0; i < k; i++){
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps * segno(m(j));
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //refraction or reflection?
        if( std::abs(m(j)) > delta_U ){
          
          //refraction
          m(j) -= segno(m(j)) * delta_U;
          U += delta_U;
          
        }else{
          
          //reflection
          theta(j) = theta_old;
          m(j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      theta.subvec(0,d-k-1) += 0.5 * eps * m.subvec(0,d-k-1);
      
      //calculate the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        
        //continuous momentum update by half step size
        m.subvec(0,d-k-1) -= 0.5 * eps * grad;
        
        //calculate the contribution to the sum of the metropolis log weights
        H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
          0.5 * arma::sum(arma::square(m.subvec(0,d-k-1))) +
          arma::sum(arma::abs(m.subvec(d-k,d-1)));
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(H1)){
          H1 = arma::datum::inf;
        }
        
        //set the acceptance probability
        //equal to the ratio between the exponential of the 2 energy levels
        accept_prob = std::exp(H0-H1);
        
      }else{
        //otherwise
        if(a == 1.0){
          //if a == 1 the last doubling was too much so we halve and start again
          eps /= 2;
          break;
        }
      }
      
    }else{
      //otherwise
      if(a == 1.0){
        //if a == 1 the last doubling was too much so we halve and start again
        eps /= 2;
        break;
      }
    }
    
  }
  
  //return a first estimate of the optimal step size
  return eps;
}


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
                    const arma::vec& M_inv_disc){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(arma::square( m0.subvec(0,d-k-1) ),M_inv_cont ) + 
    arma::dot(arma::abs(m0.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the new energy level
  double H1;
  
  //initialize the potential energy
  double U;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
  
  //calculate the value of the new potential energy
  U = Rcpp::as<double>(nlp(theta,args,true));
  
  // if the potential energy is finite then we can continue
  if(arma::is_finite(U)){
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    unsigned int j;
    
    //loop for each discontinuous coordinate
    for(unsigned int i = 0; i < k; i++){
      
      //set the index
      j = idx_disc(i);
      
      //modify the discrete parameter
      theta_old = theta(j);
      
      theta(j) = theta_old + eps * segno(m(j)) * M_inv_disc(j-d+k);
      
      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
      
      //refraction or reflection?
      if( M_inv_disc(j-d+k) * std::abs(m(j)) > delta_U ){
        
        //refraction
        m(j) -= segno(m(j)) * delta_U / M_inv_disc(j-d+k);
        U += delta_U;
        
      }else{
        
        //reflection
        theta(j) = theta_old;
        m(j) *= -1.0;
        
      }
      
    }
    
    // continue updating continuous parameters
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
    
    //calculate the gradient
    arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //let's make sure it's finished
    if(arma::is_finite(grad)){
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps * grad;
      
      //calculate the contribution to the sum of the metropolis log weights
      H1 = Rcpp::as<double>(nlp(theta,args,true)) +
        0.5 * arma::dot(arma::square( m.subvec(0,d-k-1) ),M_inv_cont ) + 
        arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!arma::is_finite(H1)){
        H1 = arma::datum::inf;
      }
      
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);
      
      //if the probability of acceptance is greater than 0.5 then
      //double the stepsize, otherwise halve it
      a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
    }
    
  }
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;

    //step forward with a modified step size
    accept_prob = 0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
    
    //calculate the value of the new potential energy
    U = Rcpp::as<double>(nlp(theta,args,true));
    
    // if the potential energy is finite then we can continue
    if(arma::is_finite(U)){
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      
      //loop for each discontinuous coordinate
      for(unsigned int i = 0; i < k; i++){
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps * segno(m(j)) * M_inv_disc(j-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //refraction or reflection?
        if( M_inv_disc(j-d+k) * std::abs(m(j)) > delta_U ){
          
          //refraction
          m(j) -= segno(m(j)) * delta_U / M_inv_disc(j-d+k);
          U += delta_U;
          
        }else{
          
          //reflection
          theta(j) = theta_old;
          m(j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
      
      //calculate the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        
        //continuous momentum update by half step size
        m.subvec(0,d-k-1) -= 0.5 * eps * grad;
        
        //calculate the contribution to the sum of the metropolis log weights
        H1 = Rcpp::as<double>(nlp(theta,args,true)) +
          0.5 * arma::dot(arma::square( m.subvec(0,d-k-1) ),M_inv_cont ) +
          arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(H1)){
          H1 = arma::datum::inf;
        }
        
        //set the acceptance probability
        //equal to the ratio between the exponential of the 2 energy levels
        accept_prob = std::exp(H0-H1);
        
      }else{
        //otherwise
        if(a == 1.0){
          //if a == 1 the last doubling was too much so we halve and start again
          eps /= 2;
          break;
        }
      }
      
    }else{
      //otherwise
      if(a == 1.0){
        //if a == 1 the last doubling was too much so we halve and start again
        eps /= 2;
        break;
      }
    }
    
  }
  
  //return a first estimate of the optimal step size
  return eps;
}


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
                    const arma::vec& M_inv_disc){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) +
    0.5 * arma::dot(m0.subvec(0,d-k-1), M_inv_cont * m0.subvec(0,d-k-1)) + 
    arma::dot(arma::abs(m0.subvec(d-k,d-1)),M_inv_disc);
  
  //initialize the new energy level
  double H1;
  
  //initialize the potential energy
  double U;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old value of the position
  double theta_old;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //continuous momentum update by half step size
  m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by half step size
  theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
  
  //calculate the value of the new potential energy
  U = Rcpp::as<double>(nlp(theta,args,true));
  
  // if the potential energy is finite then we can continue
  if(arma::is_finite(U)){
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    unsigned int j;
    
    //loop for each discontinuous coordinate
    for(unsigned int i = 0; i < k; i++){
      
      //set the index
      j = idx_disc(i);
      
      //modify the discrete parameter
      theta_old = theta(j);
      
      theta(j) = theta_old + eps * segno(m(j)) * M_inv_disc(j-d+k);
      
      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
      
      //refraction or reflection?
      if( M_inv_disc(j-d+k) * std::abs(m(j)) > delta_U ){
        
        //refraction
        m(j) -= segno(m(j)) * delta_U / M_inv_disc(j-d+k);
        U += delta_U;
        
      }else{
        
        //reflection
        theta(j) = theta_old;
        m(j) *= -1.0;
        
      }
      
    }
    
    // continue updating continuous parameters
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
    
    //calculate the gradient
    arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //let's make sure it's finished
    if(arma::is_finite(grad)){
      
      //continuous momentum update by half step size
      m.subvec(0,d-k-1) -= 0.5 * eps * grad;
      
      //calculate the contribution to the sum of the metropolis log weights
      H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
        0.5 * arma::dot(m.subvec(0,d-k-1), M_inv_cont * m.subvec(0,d-k-1)) +
        arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!arma::is_finite(H1)){
        H1 = arma::datum::inf;
      }
      
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);
      
      //if the probability of acceptance is greater than 0.5 then
      //double the stepsize, otherwise halve it
      a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
    }
    
  }
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;

    //step forward with a modified step size
    accept_prob = 0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
    
    //calculate the value of the new potential energy
    U = Rcpp::as<double>(nlp(theta,args,true));
    
    // if the potential energy is finite then we can continue
    if(arma::is_finite(U)){
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      
      //loop for each discontinuous coordinate
      for(unsigned int i = 0; i < k; i++){
        
        //set the index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps * segno(m(j)) * M_inv_disc(j-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //refraction or reflection?
        if( M_inv_disc(j-d+k) * std::abs(m(j)) > delta_U ){
          
          //refraction
          m(j) -= segno(m(j)) * delta_U / M_inv_disc(j-d+k);
          U += delta_U;
          
        }else{
          
          //reflection
          theta(j) = theta_old;
          m(j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
      
      //calculate the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        
        //continuous momentum update by half step size
        m.subvec(0,d-k-1) -= 0.5 * eps * grad;
        
        //calculate the contribution to the sum of the metropolis log weights
        H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
          0.5 * arma::dot(m.subvec(0,d-k-1), M_inv_cont * m.subvec(0,d-k-1)) + 
          arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(H1)){
          H1 = arma::datum::inf;
        }
        
        //set the acceptance probability
        //equal to the ratio between the exponential of the 2 energy levels
        accept_prob = std::exp(H0-H1);
        
      }else{
        //otherwise
        if(a == 1.0){
          //if a == 1 the last doubling was too much so we halve and start again
          eps /= 2;
          break;
        }
      }
      
    }else{
      //otherwise
      if(a == 1.0){
        //if a == 1 the last doubling was too much so we halve and start again
        eps /= 2;
        break;
      }
    }
    
  }
  
  //return a first estimate of the optimal step size
  return eps;
}


/* -------------------------------------------------------------------------- */

/* ------------------------ VERSION WITH k = 0 ------------------------------ */

/* -------------------------------------------------------------------------- */

// identity matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::sum(arma::square(m0));
  
  //initialize the new energy level
  double H1;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //continuous momentum update by half step size
  m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by one step size
  theta += eps * m;

  //continuous momentum update by half step size
  m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));;
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::sum(arma::square(m));
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //set the acceptance probability
  //equal to the ratio between the exponential of the 2 energy levels
  accept_prob = std::exp(H0-H1);
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;

    //step forward with a modified step size
    accept_prob = 0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta += eps * m;
    
    //continuous momentum update by half step size
    m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //calculate the contribution to the sum of the metropolis log weights
    H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
      0.5 * arma::sum(arma::square(m));
    
    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(arma::is_finite(H1)){
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);   

    }else{
      //otherwise

      if(a == 1.0){
        //if a == 1 the last doubling was too much so we halve and start again
        eps /= 2;
        break;
      }
    }
    
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// diagonal matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const arma::vec& M_inv){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(arma::square(m0),M_inv);
  
  //initialize the new energy level
  double H1;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //continuous momentum update by half step size
  m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by one step size
  theta += eps * M_inv % m;
  
  //continuous momentum update by half step size
  m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));;
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(arma::square(m),M_inv);
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //set the acceptance probability
  //equal to the ratio between the exponential of the 2 energy levels
  accept_prob = std::exp(H0-H1);
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;

    //step forward with a modified step size
    accept_prob = 0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta += eps * M_inv % m;
    
    //continuous momentum update by half step size
    m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //calculate the contribution to the sum of the metropolis log weights
    H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
      0.5 * arma::dot(arma::square(m),M_inv);
    
    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(arma::is_finite(H1)){
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);    
    }else{
      //otherwise
      if(a == 1.0){
        //if a == 1 the last doubling was too much so we halve and start again
        eps /= 2;
        break;
      }
    }
    
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// dense matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    const arma::mat& M_inv){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(theta0,args,true)) + 
    0.5 * arma::dot(m0,M_inv * m0);
  
  //initialize the new energy level
  double H1;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //continuous momentum update by half step size
  m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //continuous parameter update by one step size
  theta += eps * M_inv * m;
  
  //continuous momentum update by half step size
  m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));;
  
  //calculate the contribution to the sum of the metropolis log weights
  H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
    0.5 * arma::dot(m,M_inv * m);
  
  //let's make sure it's not NaN, in which case let's set it equal to -Inf
  if(!arma::is_finite(H1)){
    H1 = arma::datum::inf;
  }
  
  //set the acceptance probability
  //equal to the ratio between the exponential of the 2 energy levels
  accept_prob = std::exp(H0-H1);
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;

    //step forward with a modified step size
    accept_prob = 0;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    arma::vec theta = theta0;
    arma::vec m = m0;
    
    //let's take a leapfrog step
    
    //continuous momentum update by half step size
    m -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //continuous parameter update by half step size
    theta += eps * M_inv * m;
    
    //continuous momentum update by half step size
    m.subvec(0,d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //calculate the contribution to the sum of the metropolis log weights
    H1 = Rcpp::as<double>(nlp(theta,args,true)) + 
      0.5 * arma::dot(m,M_inv * m);
    
    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(arma::is_finite(H1)){
      //set the acceptance probability
      //equal to the ratio between the exponential of the 2 energy levels
      accept_prob = std::exp(H0-H1);    
    }else{
      //otherwise
      if(a == 1.0){
        //if a == 1 the last doubling was too much so we halve and start again
        eps /= 2;
        break;
      }
    }
    
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

/* -------------------------------------------------------------------------- */

/* ------------------------ VERSION WITH k = d ------------------------------ */

/* -------------------------------------------------------------------------- */

// identity matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    arma::uvec& idx_disc){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //initialize the current potential energy
  double U0,U;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old postition value
  double theta_old;
  
  //set to infinity the metropolis log weigths
  double lsw = -arma::datum::inf;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //calculate the value of the potential energy
  U = U0 = Rcpp::as<double>(nlp(theta,args,true));
  
  //permute the order of the discrete parameters
  idx_disc = arma::shuffle(idx_disc);
  
  unsigned int j;
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < d; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps * segno(m(j));
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(arma::is_finite(delta_U)){
      
      //update the metropolis log sum weight
      lsw = arma::log_add_exp(lsw,std::min(0.0,-delta_U));
      
      //refraction or reflection?
      if( std::abs(m(j)) > delta_U ){
        
        //refraction
        m(j) -= segno(m(j)) * delta_U;
        U += delta_U;
        
      }else{
        
        //reflection
        theta(j) = theta_old;
        m(j) *= -1.0;
        
      }
    }else{
      //set lsw equal to minus infinity
      lsw = -arma::datum::inf;
    }
    
  }
  
  //set the acceptance probability
  //equal to the average of the metropolis rates
  accept_prob = std::exp(lsw) / d;
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;

    //step forward with a modified step size
    lsw = -arma::datum::inf;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    theta = theta0;
    m = m0;
    U = U0;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    unsigned int j;
    
    //loop for each discontinuous coordinate
    for(unsigned int i = 0; i < d; i++){
      
      //set the index
      j = idx_disc(i);
      
      //modify the discrete parameter
      theta_old = theta(j);
      
      theta(j) = theta_old + eps * segno(m(j));
      
      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
      
      if(arma::is_finite(delta_U)){
        
        //update the metropolis log sum weight
        lsw = arma::log_add_exp(lsw,std::min(0.0,-delta_U));
        
        //refraction or reflection?
        if( std::abs(m(j)) > delta_U ){
          
          //refraction
          m(j) -= segno(m(j)) * delta_U;
          U += delta_U;
          
        }else{
          
          //reflection
          theta(j) = theta_old;
          m(j) *= -1.0;
          
        }
      }else{
        //otherwise
        if(a == 1.0){
          //if a == 1 the last doubling was too much so we halve and start again
          eps /= 2;
          break;
        }else{
          //set the acceptance probability to zero
          lsw = -arma::datum::inf;
        }
      }
      
    }
    //set the acceptance probability
    //equal to the average of the metropolis rates
    accept_prob = std::exp(lsw) / d;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}

// diagonal matrix case
double init_epsilon(const arma::vec& theta0,
                    const arma::vec& m0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    const unsigned int& d,
                    arma::uvec& idx_disc,
                    const arma::vec& M_inv){
  
  //initialize the current value of the step size
  double eps = 1;
  
  //initialize the current value of the acceptance probability
  double accept_prob = 0;
  
  //double or halve the stepsize?
  double a = 1.0;
  
  //initialize the current potential energy
  double U0,U;
  
  //initialize the potential difference
  double delta_U;
  
  //initialize the old postition value
  double theta_old;
  
  //set to infinity the metropolis log weigths
  double lsw = -arma::datum::inf;
  
  //initialize the current value of the position and momentum of the particle
  arma::vec theta = theta0;
  arma::vec m = m0;
  
  //let's take a leapfrog step
  
  //calculate the value of the potential energy
  U = U0 = Rcpp::as<double>(nlp(theta,args,true));
  
  //permute the order of the discrete parameters
  idx_disc = arma::shuffle(idx_disc);
  unsigned int j;
  
  //loop for each discontinuous coordinate
  for(unsigned int i = 0; i < d; i++){
    
    //set the index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps * segno(m(j)) * M_inv(j);
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    if(arma::is_finite(delta_U)){
      
      //update the metropolis log sum weight
      lsw = arma::log_add_exp(lsw,std::min(0.0,-delta_U));
      
      //refraction or reflection?
      if( M_inv(j) * std::abs(m(j)) > delta_U ){
        
        //refraction
        m(j) -= segno(m(j)) * delta_U / M_inv(j);
        U += delta_U;
        
      }else{
        
        //reflection
        theta(j) = theta_old;
        m(j) *= -1.0;
        
      }
    }else{
      //set lsw equal to minus infinity
      lsw = -arma::datum::inf;
    }
    
  }
  
  //set the acceptance probability
  //equal to the average of the metropolis rates
  accept_prob = std::exp(lsw) / d;
  
  //if the probability of acceptance is greater than 0.5 then
  //double the stepsize, otherwise halve it
  a = 2.0 * static_cast<int>(accept_prob > 0.5) - 1;
  
  //continue in this direction until the acceptance probability does not exceed 0.5
  //from the bottom or from the top depending on whether I halve or double
  
  //set the count to zero
  unsigned int count = 0;
  while( (count < 100) && ((accept_prob == 0) || (pow(accept_prob,a) > pow(0.5,a) ))){
    //update the count
    count++;
    
    //step forward with a modified step size
    lsw = -arma::datum::inf;
    
    //double or halve the step size
    eps *= pow(2,a);
    
    //initialize the current value of the position and momentum of the particle
    theta = theta0;
    m = m0;
    U = U0;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    unsigned int j;
    //loop for each discontinuous coordinate
    for(unsigned int i = 0; i < d; i++){
      
      //set the index
      j = idx_disc(i);
      
      //modify the discrete parameter
      theta_old = theta(j);
      
      theta(j) = theta_old + eps * segno(m(j)) * M_inv(j);
      
      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
      
      if(arma::is_finite(delta_U)){
        
        //update the metropolis log sum weight
        lsw = arma::log_add_exp(lsw,std::min(0.0,-delta_U));
        
        //refraction or reflection?
        if( M_inv(j) * std::abs(m(j)) > delta_U ){
          
          //refraction
          m(j) -= segno(m(j)) * delta_U / M_inv(j);
          U += delta_U;
          
        }else{
          
          //reflection
          theta(j) = theta_old;
          m(j) *= -1.0;
          
        }
      }else{
        //otherwise
        if(a == 1.0){
          //if a == 1 the last doubling was too much so we halve and start again
          eps /= 2;
          break;
        }else{
          //set the acceptance probability to zero
          lsw = -arma::datum::inf;
        }
      }
      
    }
    //set the acceptance probability
    //equal to the average of the metropolis rates
    accept_prob = std::exp(lsw) / d;
  }
  
  //return a first estimate of the optimal step size
  return eps;
}
