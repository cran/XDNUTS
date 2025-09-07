#include <iostream>
#include <RcppArmadillo.h>
#include <cmath>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "leapfrog.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS FOR MAKE A LEAPFROG STEP

//identity matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const unsigned int& k,
              arma::uvec idx_disc){
  
  //set the metropolis log sum weights sum to minus infinity
  lsw = -arma::datum::inf;
  
  //compute the gradient
  arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //if the gradient is finite we can continue
  if(grad.is_finite()){
    
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * grad;
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * m.subvec(0,d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(theta,args,true));
    
    // if the potential energy is finite then we continue
    if(std::isfinite(U)){
      
      //initialization of the new vector and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps * segno(m(j));
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        alpha(1+j-d+k) += std::min(1.0,std::exp(-delta_U));
        
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
      
      //compute the gradient
      grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //let's make sure it's finite
      if(grad.is_finite()){
        //continuous momentum update by half step size
        m.subvec(0,d-k-1) -= 0.5 * eps * grad;
        
        //compute the contribution to the sum of the log weights metropolis
        lsw = -Rcpp::as<double>(nlp(theta,args,true)) - 
          0.5*arma::sum(arma::square(m.subvec(0 ,d-k-1))) - 
          arma::sum(arma::abs(m.subvec(d-k,d-1)));
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!std::isfinite(lsw)){
          lsw = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-lsw - H0) > 1000){
          
          //add the divergent transition to the global matrix
          theta.subvec(0,d-k-1) -= 0.5 * eps * m.subvec(0,d-k-1);
          add_div_trans(theta.subvec(0,d-1));
          
        }else{
          //update the metropolis acceptance rate
          alpha(0) += std::min(1.0,std::exp(H0+lsw));
        }
      }else{

        //add the divergent transition to the global matrix
        theta.subvec(0,d-k-1) -= 0.5 * eps * m.subvec(0,d-k-1);
        add_div_trans(theta.subvec(0,d-1));
      }
    }else{

      //add the divergent transition to the global matrix
      theta.subvec(0,d-k-1) -= 0.5 * eps * m.subvec(0,d-k-1);
      add_div_trans(theta.subvec(0,d-1));
    }
  }else{
    //add the divergent transition to the global matrix
    add_div_trans(theta.subvec(0,d-1));
  }
  
}

//diagonal matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const unsigned int& k,
              arma::uvec idx_disc,
              const arma::vec& M_inv_cont,
              const arma::vec& M_inv_disc){
  
  //set the metropolis log sum weights sum to minus infinity
  lsw = -arma::datum::inf;

  //compute the gradient
  arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //if the gradient is finite we can continue
  if(grad.is_finite()){
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * grad;
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(theta,args,true));
    
    // if the potential energy is finite then we continue
    if(std::isfinite(U)){
      
      //initialization of the new vector and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps * segno(m(j)) * M_inv_disc(j-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        alpha(1+j-d+k) += std::min(1.0,std::exp(-delta_U));
        
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
      
      //compute the gradient
      grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //let's make sure it's finite
      if(grad.is_finite()){
        //continuous momentum update by half step size
        m.subvec(0,d-k-1) -= 0.5 * eps * grad;
        
        //compute the contribution to the sum of the log weights metropolis
        lsw = -Rcpp::as<double>(nlp(theta,args,true)) - 
          0.5*arma::dot(arma::square(m.subvec(0 ,d-k-1)),M_inv_cont) - 
          arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!std::isfinite(lsw)){
          lsw = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-lsw - H0) > 1000){
          
          //add the divergent transition to the global matrix
          theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
          add_div_trans(theta.subvec(0,d-1));
          
        }else{
          //update the metropolis acceptance rate
          alpha(0) += std::min(1.0,std::exp(H0+lsw));
        }
      }else{
        
        //add the divergent transition to the global matrix
        theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
        add_div_trans(theta.subvec(0,d-1));
      }
    }else{

      //add the divergent transition to the global matrix
      theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
      add_div_trans(theta.subvec(0,d-1));
    }
  }else{
    //add the divergent transition to the global matrix
    add_div_trans(theta.subvec(0,d-1));
  }
  
}

//dense matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const unsigned int& k,
              arma::uvec idx_disc,
              const arma::mat& M_inv_cont,
              const arma::vec& M_inv_disc){
  
  //set the metropolis log sum weights sum to minus infinity
  lsw = -arma::datum::inf;
  
  //compute the gradient
  arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //if the gradient is finite we can continue
  if(grad.is_finite()){
    //continuous momentum update by half step size
    m.subvec(0,d-k-1) -= 0.5 * eps * grad;
    
    //continuous parameter update by half step size
    theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(theta,args,true));
    
    // if the potential energy is finite then we continue
    if(std::isfinite(U)){
      
      //initialization of the new vector and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx_disc(i);
        
        //modify the discrete parameter
        theta_old = theta(j);
        
        theta(j) = theta_old + eps * segno(m(j)) * M_inv_disc(j-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        alpha(1+j-d+k) += std::min(1.0,std::exp(-delta_U));
        
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
      
      //compute the gradient
      grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
      
      //let's make sure it's finite
      if(grad.is_finite()){
        //continuous momentum update by half step size
        m.subvec(0,d-k-1) -= 0.5 * eps * grad;
        
        //compute the contribution to the sum of the log weights metropolis
        lsw = -Rcpp::as<double>(nlp(theta,args,true)) - 
          0.5*arma::dot(m.subvec(0 ,d-k-1),M_inv_cont * m.subvec(0 ,d-k-1)) - 
          arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!std::isfinite(lsw)){
          lsw = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-lsw - H0) > 1000){
          
          //add the divergent transition to the global matrix
          theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
          add_div_trans(theta.subvec(0,d-1));
          
        }else{
          //update the metropolis acceptance rate
          alpha(0) += std::min(1.0,std::exp(H0+lsw));
        }
      }else{
        
        //add the divergent transition to the global matrix
        theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
        add_div_trans(theta.subvec(0,d-1));
      }
    }else{
      
      //add the divergent transition to the global matrix
      theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
      add_div_trans(theta.subvec(0,d-1));
    }
  }else{
    //add the divergent transition to the global matrix
    add_div_trans(theta.subvec(0,d-1));
  }
}

/* --------------------------------- CASE K = 0 ----------------------------- */

//identity matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d){
  
  //set the metropolis log sum weights sum to minus infinity
  lsw = -arma::datum::inf;
  
  //compute the gradient
  arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //if the gradient is finite we can continue
  if(grad.is_finite()){
    //continuous momentum update by half step size
    m -= 0.5 * eps * grad;
    
    //continuous parameter update by half step size
    theta += eps * m;
    
    //compute the gradient
    grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //if the gradient is finite we can continue
    if(grad.is_finite()){
      //continuous momentum update by half step size
      m -= 0.5 * eps * grad;
      
      //compute the contribution to the sum of the log weights metropolis
      lsw = -Rcpp::as<double>(nlp(theta,args,true)) - 
        0.5*arma::sum(arma::square(m)); 
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!std::isfinite(lsw)){
        lsw = -arma::datum::inf;
      }
      
      //let's check if there is a divergent transition
      if( (-lsw - H0) > 1000){
        
        //add the divergent transition to the global matrix
        theta -= 0.5 * eps * m;
        add_div_trans(theta.subvec(0,d-1));
        
      }else{
        //update the metropolis acceptance rate
        alpha(0) += std::min(1.0,std::exp(H0+lsw));
      }
    }else{
      //add the divergent transition to the global matrix
      theta -= 0.5 * eps * m;
      add_div_trans(theta.subvec(0,d-1));    
    }
    
  }else{
    //add the divergent transition to the global matrix
    add_div_trans(theta.subvec(0,d-1));
  }
  
}

//diagonal matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const arma::vec& M_inv){
  
  //set the metropolis log sum weights sum to minus infinity
  lsw = -arma::datum::inf;
  
  //compute the gradient
  arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //if the gradient is finite we can continue
  if(grad.is_finite()){
    //continuous momentum update by half step size
    m -= 0.5 * eps * grad;
    
    //continuous parameter update by half step size
    theta += eps * M_inv % m;
    
    //compute the gradient
    grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //if the gradient is finite we can continue
    if(grad.is_finite()){
      //continuous momentum update by half step size
      m -= 0.5 * eps * grad;
      
      //compute the contribution to the sum of the log weights metropolis
      lsw = -Rcpp::as<double>(nlp(theta,args,true)) - 
        0.5*arma::dot(arma::square(m),M_inv); 
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!std::isfinite(lsw)){
        lsw = -arma::datum::inf;
      }
      
      //let's check if there is a divergent transition
      if( (-lsw - H0) > 1000){
        
        //add the divergent transition to the global matrix
        theta -= 0.5 * eps * M_inv % m;
        add_div_trans(theta.subvec(0,d-1));
        
      }else{
        //update the metropolis acceptance rate
        alpha(0) += std::min(1.0,std::exp(H0+lsw));
      }
    }else{
      //add the divergent transition to the global matrix
      theta -= 0.5 * eps * M_inv % m;
      add_div_trans(theta.subvec(0,d-1));
    }
    
  }else{
    //add the divergent transition to the global matrix
    add_div_trans(theta.subvec(0,d-1));
  }
  
}

//dense matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              double& lsw,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              const double& H0,
              const unsigned int& d,
              const arma::mat& M_inv){
  
  //set the metropolis log sum weights sum to minus infinity
  lsw = -arma::datum::inf;
  
  //compute the gradient
  arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
  
  //if the gradient is finite we can continue
  if(grad.is_finite()){
    //continuous momentum update by half step size
    m -= 0.5 * eps * grad;
    
    //continuous parameter update by half step size
    theta += eps * M_inv * m;
    
    //compute the gradient
    grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
    
    //if the gradient is finite we can continue
    if(grad.is_finite()){
      //continuous momentum update by half step size
      m -= 0.5 * eps * grad;
      
      //compute the contribution to the sum of the log weights metropolis
      lsw = -Rcpp::as<double>(nlp(theta,args,true)) - 
        0.5*arma::dot(m,M_inv * m); 
      
      //let's make sure it's not NaN, in which case let's set it equal to -Inf
      if(!std::isfinite(lsw)){
        lsw = -arma::datum::inf;
      }
      
      //let's check if there is a divergent transition
      if( (-lsw - H0) > 1000){

        //add the divergent transition to the global matrix
        theta -= 0.5 * eps * M_inv * m;
        add_div_trans(theta.subvec(0,d-1));
        
      }else{
        //update the metropolis acceptance rate
        alpha(0) += std::min(1.0,std::exp(H0+lsw));
      }
    }else{
      //add the divergent transition to the global matrix
      theta -= 0.5 * eps * M_inv * m;
      add_div_trans(theta.subvec(0,d-1));
    }
    
  }else{
    //add the divergent transition to the global matrix
    add_div_trans(theta.subvec(0,d-1));
  }

}

/* ------------------------------ CASE K = D -------------------------------- */

//identity matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              double& U,
              const unsigned int& d,
              arma::uvec& idx_disc){
  
  //initialization of the new vector and the potential difference
  double theta_old;
  double delta_U;
  
  //permute the order of the discrete parameters
  idx_disc = arma::shuffle(idx_disc);
  
  unsigned int j;
  //loop for every discontinuous component
  for(unsigned int i = 0; i < d; i++){
    
    //set the current index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps * segno(m(j));
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    //let's make sure it's finite
    if(std::isnan(delta_U)){
      delta_U = -arma::datum::inf;
      theta(j) = theta_old;
      //add the divergent transition to the global matrix
      add_div_trans(theta.subvec(0,d-1));
      
      break;
      
    }else{
      //calculation of the Metropolis acceptance rate
      alpha(j) += std::min(1.0,std::exp(-delta_U));
      
      //refraction or reflection?
      if( std::abs(m(j)) > delta_U ){
        
        //refraction
        m(j) -= segno(m(j)) * delta_U;
        U += delta_U;
        
        //Rcpp::Rcout << "sub_tree" << sub_tree.t() << std::endl;
        
      }else{
        
        //reflection
        theta(j) = theta_old;
        m(j) *= -1.0;
        
      }
    }
  }
}

//diagonal matrix case
void leapfrog(arma::vec& theta,
              arma::vec& m,
              arma::vec& alpha,
              const double& eps,
              const Rcpp::Function& nlp,
              const Rcpp::List& args,
              double& U,
              const unsigned int& d,
              arma::uvec& idx_disc,
              const arma::vec& M_inv){
  
  //initialization of the new vector and the potential difference
  double theta_old;
  double delta_U;
  
  //permute the order of the discrete parameters
  idx_disc = arma::shuffle(idx_disc);
  
  unsigned int j;
  //loop for every discontinuous component
  for(unsigned int i = 0; i < d; i++){
    
    //set the current index
    j = idx_disc(i);
    
    //modify the discrete parameter
    theta_old = theta(j);
    
    theta(j) = theta_old + eps * segno(m(j)) * M_inv(j);
    
    //calculation of the difference in potential energy
    delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
    
    //let's make sure it's finite
    if(std::isnan(delta_U)){
      delta_U = -arma::datum::inf;
      theta(j) = theta_old;
      //add the divergent transition to the global matrix
      add_div_trans(theta.subvec(0,d-1));
      
      break;
      
    }else{
      //calculation of the Metropolis acceptance rate
      alpha(j) += std::min(1.0,std::exp(-delta_U));
      
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
    }
  }
}
