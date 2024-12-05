#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "recursive_tree.h"
#include "leapfrog.h"
#include "single_hmc.h"
#include "single_nuts.h"
#include "epsilon_init.h"
#include "epsilon_adapt.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// FUNCTION FOR STEP SIZE CALIBRATION

// identity matrix case
void adapt_stepsize(arma::vec& theta0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    double& eps0,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const Rcpp::List& control,
                    const unsigned int& N,
                    const double& log_tau,
                    const unsigned int& L){
  
  //set the log value on which the step size is shrunked
  double mu = std::log(10*eps0);
  
  //initialize the log value of the current step size
  double l_eps_init = std::log(eps0);
  
  //initialize the moving average of the log step size value
  double l_eps_bar = 0.0;
  
  //initialize the moving average of the statistic to be brought to zero
  double Hbar = 0.0;
  
  //additional dummy time to prevent the first iterations from being too intrusive
  double t0 = Rcpp::as<double>(control["t0"]);
  
  //metropolis acceptance rates you want to achieve
  arma::vec delta = Rcpp::as<arma::vec>(control["delta"]);
  
  //parameter that regulates the influence of updates
  double gamma = Rcpp::as<double>(control["gamma"]);
  
  //parameter that makes the procedure venescent
  double kappa = Rcpp::as<double>(control["kappa"]);
  
  //maximum depth of the binary tree
  unsigned int max_treedepth = Rcpp::as<unsigned int>(control["max_treedepth"]);
  
  //initialize the current value of the particle's position in space
  arma::vec theta = theta0;
  arma::vec m(theta.size());
  
  //initialize the list containing the output of one chain iteration
  Rcpp::List iteration;
  
  //value of the acceptance rate of an iteration
  arma::vec alpha;
  
  //divergence of this value from the nominal one
  double gain = 0;
  
  //distinguish the three algorithms
  if(log_tau == 1000 && L == 0){
    //dnuts
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a nuts iteration
      if(k == 0){
        //classic nuts
        
        //generate particle momentum
        m = arma::randn(d);
        
        //nuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init),
                                 max_treedepth,
                                 d);
      }else if(k == d){
        //pure dnuts
        
        //generate particle momentum
        m = rlaplace(d);
        
        //dnuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init),
                                 max_treedepth,
                                 d,
                                 idx_disc);
      }else {
        //mixed dnuts
        
        //generate particle momenta
        m.subvec(0,d-k-1) = arma::randn(d-k);
        m.subvec(d-k,d-1) = rlaplace(k);
        
        //dnuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      if(k == d){
        //pure dnuts case
        gain = delta(1) - arma::mean(alpha);
        if(!arma::is_finite(gain)){
          gain = delta(1);
        }
      }else if(k == 0 || std::isnan(delta(1))){
        
        //clasic nuts case
        gain = delta(0) - alpha(0); 
        if(!arma::is_finite(gain)){
          gain = delta(0);
        }
      }else{
        //mixed dnuts case
        gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); //+ arma::var(alpha.subvec(1,k));
        if(!arma::is_finite(gain)){
          gain = delta(0);
        }
      }
      
      //update the moving average of the statistic to be brought to zero
      Hbar = (1 - 1/(t0+i))*Hbar + gain / (t0 + i);
      
      //update the estimate of the logarithm of the step size
      l_eps_init = mu - Hbar * std::sqrt(static_cast<double>(i)) / gamma;
      
      //update the moving average of the log value of the step size
      l_eps_bar = (pow(i,-kappa)) * l_eps_init + (1.0 - pow(i,-kappa)) * l_eps_bar;
      
    }
  }else if(L == 0){
    //xdhmc
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a xhmc iteration 
      if(k == 0){
        //classic xhmc
        
        //generate particle momentum
        m = arma::randn(d);
        
        //xhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init),
                                 max_treedepth,
                                 d,
                                 log_tau);
      }else if(k == d){
        //pure dxhmc
        
        //generate particle momentum
        m = rlaplace(d);
        
        //xdhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 log_tau);
      }else {
        //mixed dxhmc
        
        //generate particle momentum
        m.subvec(0,d-k-1) = arma::randn(d-k);
        m.subvec(d-k,d-1) = rlaplace(k);
        
        //xdhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 log_tau);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);

      if(k == d){
        //pure xdhmc case
        gain = delta(1) - arma::mean(alpha);
        if(!arma::is_finite(gain)){
          gain = delta(1);
        }
      }else if(k == 0 || std::isnan(delta(1))){
        //xhmc
        
        
        gain = delta(0) - alpha(0); 
        if(!arma::is_finite(gain)){
          gain = delta(0);
        }
      }else{
        //xdhmc
        gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); //+ arma::var(alpha.subvec(1,k));
        if(!arma::is_finite(gain)){
          gain = delta(0);
        }
      }
      
      //update the moving average of the statistic to be brought to zero
      Hbar = (1 - 1/(t0+i))*Hbar + gain / (t0 + i);
      
      //update the estimate of the logarithm of the step size
      l_eps_init = mu - Hbar * std::sqrt(static_cast<double>(i)) / gamma;
      
      //update the moving average of the log value of the step size
      l_eps_bar = (pow(i,-kappa)) * l_eps_init + (1.0 - pow(i,-kappa)) * l_eps_bar;
      
    }
  }else{
    //dhmc
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a hmc iteration
      if(k == 0){
        //classic hmc
        
        //generate particle momentum
        m = arma::randn(d);
        
        //hmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init),
                                max_treedepth,
                                d);
      }else if(k == d){
        //pure dhmc
        
        //generate particle momentum
        m = rlaplace(d);
        
        //dhmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init),
                                max_treedepth,
                                d,
                                idx_disc);
      }else {
        //dhmc
        
        //generate particle momentum
        m.subvec(0,d-k-1) = arma::randn(d-k);
        m.subvec(d-k,d-1) = rlaplace(k);
        
        //dhmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init),
                                max_treedepth,
                                d,
                                k,
                                idx_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      if(k == d){
        //pure dhmc case
        gain = delta(1) - arma::mean(alpha);
        if(!arma::is_finite(gain)){
          gain = delta(1);
        }
      }else if(k == 0 || std::isnan(delta(1))){
        //hmc
        
        
        gain = delta(0) - alpha(0); 
        if(!arma::is_finite(gain)){
          gain = delta(0);
        }
      }else{
        //dhmc
        gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); //+ arma::var(alpha.subvec(1,k));
        if(!arma::is_finite(gain)){
          gain = delta(0);
        }
      }
      
      //update the moving average of the statistic to be brought to zero
      Hbar = (1 - 1/(t0+i))*Hbar + gain / (t0 + i);
      
      //update the estimate of the logarithm of the step size
      l_eps_init = mu - Hbar * std::sqrt(static_cast<double>(i)) / gamma;
      
      //update the moving average of the log value of the step size
      l_eps_bar = (pow(i,-kappa)) * l_eps_init + (1.0 - pow(i,-kappa)) * l_eps_bar;
      
    }
  }
  
  
  //update position value and step size 
  theta0 = theta;
  eps0 = std::exp(l_eps_bar);
}


/* -------------------------------------------------------------------------- */


// diagonal matrix case
void adapt_stepsize(arma::vec& theta0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    double& eps0,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const Rcpp::List& control,
                    const unsigned int& N,
                    const double& log_tau,
                    const unsigned int& L,
                    const arma::vec& M_cont,
                    arma::vec& M_disc,
                    const arma::vec& M_inv_cont,
                    arma::vec& M_inv_disc){
  
  //check the condition of different stepsize
  bool condition = Rcpp::as<bool>(control["different_stepsize"]) && (k != 0);
  
  //set the log value on which the step size is shrunked
  arma::vec mu(k+1);
  mu(0) = std::log(10*eps0);
  if(condition){
    mu.subvec(1,k) = arma::log(10*eps0* M_inv_disc);
  }
  
  //initialize the log value of the current step size
  arma::vec l_eps_init(k+1);
  l_eps_init(0) = std::log(eps0);
  
  //initialize the different step size
  if(condition){
    l_eps_init.subvec(1,k) = arma::log(eps0*M_inv_disc);
  }
  
  //initialize the moving average of the log step size value
  arma::vec l_eps_bar = arma::zeros<arma::vec>(k+1);
  
  //initialize the moving average of the statistic to be brought to zero
  arma::vec Hbar = arma::zeros<arma::vec>(k+1);
  
  //additional dummy time to prevent the first iterations from being too intrusive
  double t0 = Rcpp::as<double>(control["t0"]);
  
  //metropolis acceptance rates you want to achieve
  arma::vec delta = Rcpp::as<arma::vec>(control["delta"]);
  
  //parameter that regulates the influence of updates
  double gamma = Rcpp::as<double>(control["gamma"]);
  
  //parameter that makes the procedure venescent
  double kappa = Rcpp::as<double>(control["kappa"]);
  
  //maximum depth of the binary tree
  unsigned int max_treedepth = Rcpp::as<unsigned int>(control["max_treedepth"]);
  
  //index
  unsigned int j = 0;
  
  //starting point
  unsigned int start = 1;
  
  //initialize the current value of the particle's position in space
  arma::vec theta = theta0;
  arma::vec m(theta.size());
  
  //initialize the list containing the output of one chain iteration
  Rcpp::List iteration;
  
  //value of the acceptance rate of an iteration
  arma::vec alpha;
  
  //divergence of this value from the nominal one
  double gain = 0;
  
  //distinguish the three algorithms
  if(log_tau == 1000 && L == 0){
    //dnuts
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a nuts iteration
      if(k == 0){
        //classic nuts
        
        //generate particle momentum
        m = M_cont % arma::randn(d);
        
        //nuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 M_inv_cont);
        
      }else if(k == d){
        //pure dnuts
        
        //generate particle momentum
        m = M_disc % rlaplace(d);
        
        //dnuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 M_inv_disc);
      }else {
        //mixed dnuts
        
        //generate particle momenta
        m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //dnuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 M_inv_cont,
                                 M_inv_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      //check condition for different stepsize
      if(condition){
        //different step size achieved by updating the discontinuous Mass Matrix
        
        //updating the global step size
        
        if(k != d){
          //global gain
          gain = delta(0) - alpha(0);
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
          
        }else{
          start = 0;
        }
        
        
        //updating the single discontinuous components Mass Matrix 
        for(j = 0; j < k; j++){
          
          //local gain
          gain = delta(1) - alpha(start+j);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(1+j) = (1 - 1/(t0+i))*Hbar(1+j) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(1+j) = mu(1+j) - Hbar(1+j) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(1+j) = (pow(i,-kappa)) * l_eps_init(1+j) + (1.0 - pow(i,-kappa)) * l_eps_bar(1+j);
          
          //update the mass matrix
          M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
          //M_disc(j) = 1.0 / M_inv_disc(j);
          
        }
        
        //updating the global and local step size once more if k == d
        if(k == d){
          l_eps_init(0) = arma::min(l_eps_init.subvec(1,k));
          l_eps_bar(0) = arma::min(l_eps_bar.subvec(1,k));
          
          //reupdate the mass matrix
          for(j = 0; j < k; j++){
            M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
            //M_disc(j) = 1.0 / M_inv_disc(j);
          }
        }
        
      }else{
        if(k == d){
          //pure dnuts case
          gain = delta(1) - arma::mean(alpha);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
        }else if(k == 0 || std::isnan(delta(1))){
          //clasic nuts case
          
          gain = delta(0) - alpha(0); 
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }else{
          //mixed dnuts case
          gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); 
          //+ arma::var(alpha.subvec(1,k));
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }
        
        //update the moving average of the statistic to be brought to zero
        Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
        
        //update the estimate of the logarithm of the step size
        l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
        
        //update the moving average of the log value of the step size
        l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
      }
      
      
    }
    
  }else if(L == 0){
    //xdhmc
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a xhmc iteration
      if(k == 0){
        //classic xhmc
        
        //generate particle momentum
        m = M_cont % arma::randn(d);
        
        //xhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 log_tau,
                                 M_inv_cont);
        
      }else if(k == d){
        //pure dxhmc
        
        //generate particle momentum
        m = M_disc % rlaplace(d);
        
        //xdhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 log_tau,
                                 M_inv_disc);
      }else {
        //mixed dxhmc

        //generate particle momentum
        m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //xdhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 log_tau,
                                 M_inv_cont,
                                 M_inv_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      //check condition for different stepsize
      if(condition){
        //different step size achieved by updating the discontinuous Mass Matrix
        
        //updating the global step size
        
        if(k != d){
          //global gain
          gain = delta(0) - alpha(0);
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
          
        }else{
          start = 0;
        }
        
        
        //updating the single discontinuous components Mass Matrix 
        for(j = 0; j < k; j++){
          
          //local gain
          gain = delta(1) - alpha(start+j);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(1+j) = (1 - 1/(t0+i))*Hbar(1+j) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(1+j) = mu(1+j) - Hbar(1+j) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(1+j) = (pow(i,-kappa)) * l_eps_init(1+j) + (1.0 - pow(i,-kappa)) * l_eps_bar(1+j);
          
          //update the mass matrix
          M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
          //M_disc(j) = 1.0 / M_inv_disc(j);
          
        }
        
        //updating the global and local step size once more if k == d
        if(k == d){
          l_eps_init(0) = arma::min(l_eps_init.subvec(1,k));
          l_eps_bar(0) = arma::min(l_eps_bar.subvec(1,k));
          
          //reupdate the mass matrix
          for(j = 0; j < k; j++){
            M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
            //M_disc(j) = 1.0 / M_inv_disc(j);
          }
        }
        
      }else{
        if(k == d){
          //pure dnuts case
          gain = delta(1) - arma::mean(alpha);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
        }else if(k == 0 || std::isnan(delta(1))){
          //clasic nuts case
          
          gain = delta(0) - alpha(0); 
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }else{
          //mixed dnuts case
          gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); 
          //+ arma::var(alpha.subvec(1,k));
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }
        
        //update the moving average of the statistic to be brought to zero
        Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
        
        //update the estimate of the logarithm of the step size
        l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
        
        //update the moving average of the log value of the step size
        l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
      }
      
    }
  }else{
    //dhmc
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a hmc iteration
      if(k == 0){
        //classic hmc
        
        //generate particle momentum
        m = M_cont % arma::randn(d);
        
        //hmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init(0)),
                                max_treedepth,
                                d,
                                M_inv_cont);
        
      }else if(k == d){
        //pure dhmc
        
        //generate particle momentum
        m = M_disc % rlaplace(d);
        
        //dhmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init(0)),
                                max_treedepth,
                                d,
                                idx_disc,
                                M_inv_disc);
      }else {
        //dhmc
        
        //generate particle momentum
        m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //dhmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init(0)),
                                max_treedepth,
                                d,
                                k,
                                idx_disc,
                                M_inv_cont,
                                M_inv_disc);
        
      }
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      //check condition for different stepsize
      if(condition){
        //different step size achieved by updating the discontinuous Mass Matrix
        
        //updating the global step size
        
        if(k != d){
          //global gain
          gain = delta(0) - alpha(0);
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
          
        }else{
          start = 0;
        }
        
        
        //updating the single discontinuous components Mass Matrix 
        for(j = 0; j < k; j++){
          
          //local gain
          gain = delta(1) - alpha(start+j);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(1+j) = (1 - 1/(t0+i))*Hbar(1+j) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(1+j) = mu(1+j) - Hbar(1+j) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(1+j) = (pow(i,-kappa)) * l_eps_init(1+j) + (1.0 - pow(i,-kappa)) * l_eps_bar(1+j);
          
          //update the mass matrix
          M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
          //M_disc(j) = 1.0 / M_inv_disc(j);
          
        }
        
        //updating the global and local step size once more if k == d
        if(k == d){
          l_eps_init(0) = arma::min(l_eps_init.subvec(1,k));
          l_eps_bar(0) = arma::min(l_eps_bar.subvec(1,k));
          
          //reupdate the mass matrix
          for(j = 0; j < k; j++){
            M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
            //M_disc(j) = 1.0 / M_inv_disc(j);
          }
        }
        
      }else{
        if(k == d){
          //pure dnuts case
          gain = delta(1) - arma::mean(alpha);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
        }else if(k == 0 || std::isnan(delta(1))){
          //clasic nuts case
          
          gain = delta(0) - alpha(0); 
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }else{
          //mixed dnuts case
          gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); 
          //+ arma::var(alpha.subvec(1,k));
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }
        
        //update the moving average of the statistic to be brought to zero
        Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
        
        //update the estimate of the logarithm of the step size
        l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
        
        //update the moving average of the log value of the step size
        l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
      }
      
    }
  }
  
  
  //update position value and step size 
  theta0 = theta;
  eps0 = std::exp(l_eps_bar(0));
  
  //for different step size also update the discontinuous components mass matrix
  if(condition){
    for(j = 0; j < k; j++){
      M_inv_disc(j) = std::exp(l_eps_bar(1 + j) - l_eps_bar(0));
      M_disc(j) = 1.0 / M_inv_disc(j);
    }
  }
}



/* -------------------------------------------------------------------------- */


// dense matrix case
void adapt_stepsize(arma::vec& theta0,
                    const Rcpp::Function& nlp,
                    const Rcpp::List& args,
                    double& eps0,
                    const unsigned int& d,
                    const unsigned int& k,
                    arma::uvec& idx_disc,
                    const Rcpp::List& control,
                    const unsigned int& N,
                    const double& log_tau,
                    const unsigned int& L,
                    const arma::mat& M_cont,
                    arma::vec& M_disc,
                    const arma::mat& M_inv_cont,
                    arma::vec& M_inv_disc){
  
  //check the condition of different stepsize
  bool condition = Rcpp::as<bool>(control["different_stepsize"]) && (k != 0);
  
  //set the log value on which the step size is shrunked
  arma::vec mu(k+1);
  mu(0) = std::log(10*eps0);
  if(condition){
    mu.subvec(1,k) = arma::log(10*eps0 * M_inv_disc);
  }
  
  //initialize the log value of the current step size
  arma::vec l_eps_init(k+1);
  l_eps_init(0) = std::log(eps0);
  
  //initialize the different step size
  if(condition){
    l_eps_init.subvec(1,k) = arma::log(10*eps0 * M_inv_disc);
  }
  
  //initialize the moving average of the log step size value
  arma::vec l_eps_bar = arma::zeros<arma::vec>(k+1);
  
  //initialize the moving average of the statistic to be brought to zero
  arma::vec Hbar = arma::zeros<arma::vec>(k+1);
  
  //additional dummy time to prevent the first iterations from being too intrusive
  double t0 = Rcpp::as<double>(control["t0"]);
  
  //metropolis acceptance rates you want to achieve
  arma::vec delta = Rcpp::as<arma::vec>(control["delta"]);
  
  //parameter that regulates the influence of updates
  double gamma = Rcpp::as<double>(control["gamma"]);
  
  //parameter that makes the procedure venescent
  double kappa = Rcpp::as<double>(control["kappa"]);
  
  //maximum depth of the binary tree
  unsigned int max_treedepth = Rcpp::as<unsigned int>(control["max_treedepth"]);
  
  //index
  unsigned int j = 0;
  
  //starting index
  unsigned int start = 1;
  
  //initialize the current value of the particle's position in space
  arma::vec theta = theta0;
  arma::vec m(theta.size());
  
  //initialize the list containing the output of one chain iteration
  Rcpp::List iteration;
  
  //value of the acceptance rate of an iteration
  arma::vec alpha;
  
  //divergence of this value from the nominal one
  double gain = 0;
  
  //distinguish the three algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a nuts iteration
      if(k == 0){
        //classic nuts
        
        //generate particle momentum
        m = M_cont * arma::randn(d);
        
        //nuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 M_inv_cont);
        
      }else if(k == d){
        //pure dnuts
        
        //generate particle momentum
        m = M_disc % rlaplace(d);
        
        //dnuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 M_inv_disc);
      }else {
        //mixed dnuts
        
        //generate particle momenta
        m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //dnuts iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 M_inv_cont,
                                 M_inv_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      //check condition for different stepsize
      if(condition){
        //different step size achieved by updating the discontinuous Mass Matrix
        
        //updating the global step size
        
        if(k != d){
          //global gain
          gain = delta(0) - alpha(0);
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
          
        }else{
          start = 0;
        }
        
        
        //updating the single discontinuous components Mass Matrix 
        for(j = 0; j < k; j++){
          
          //local gain
          gain = delta(1) - alpha(start+j);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(1+j) = (1 - 1/(t0+i))*Hbar(1+j) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(1+j) = mu(1+j) - Hbar(1+j) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(1+j) = (pow(i,-kappa)) * l_eps_init(1+j) + (1.0 - pow(i,-kappa)) * l_eps_bar(1+j);
          
          //update the mass matrix
          M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
          //M_disc(j) = 1.0 / M_inv_disc(j);
          
        }
        
        //updating the global and local step size once more if k == d
        if(k == d){
          l_eps_init(0) = arma::min(l_eps_init.subvec(1,k));
          l_eps_bar(0) = arma::min(l_eps_bar.subvec(1,k));
          
          //reupdate the mass matrix
          for(j = 0; j < k; j++){
            M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
            //M_disc(j) = 1.0 / M_inv_disc(j);
          }
        }
        
      }else{
        if(k == d){
          //pure dnuts case
          gain = delta(1) - arma::mean(alpha);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
        }else if(k == 0 || std::isnan(delta(1))){
          //clasic nuts case
          
          gain = delta(0) - alpha(0); 
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }else{
          //mixed dnuts case
          gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); 
          //+ arma::var(alpha.subvec(1,k));
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }
        
        //update the moving average of the statistic to be brought to zero
        Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
        
        //update the estimate of the logarithm of the step size
        l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
        
        //update the moving average of the log value of the step size
        l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
      }
      
    }
  }else if(L == 0){
    //xdhmc
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a xhmc iteration
      if(k == 0){
        //classic xhmc
        
        //generate particle momentum
        m = M_cont * arma::randn(d);
        
        //xhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 log_tau,
                                 M_inv_cont);
        
      }else if(k == d){
        //pure dxhmc
        
        //generate particle momentum
        m = M_disc % rlaplace(d);
        
        //xdhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 log_tau,
                                 M_inv_disc);
      }else {
        //mixed dxhmc
        
        //generate particle momentum
        m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //xdhmc iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 std::exp(l_eps_init(0)),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 log_tau,
                                 M_inv_cont,
                                 M_inv_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      //check condition for different stepsize
      if(condition){
        //different step size achieved by updating the discontinuous Mass Matrix
        
        //updating the global step size
        
        if(k != d){
          //global gain
          gain = delta(0) - alpha(0);
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
          
        }else{
          start = 0;
        }
        
        
        //updating the single discontinuous components Mass Matrix 
        for(j = 0; j < k; j++){
          
          //local gain
          gain = delta(1) - alpha(start+j);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(1+j) = (1 - 1/(t0+i))*Hbar(1+j) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(1+j) = mu(1+j) - Hbar(1+j) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(1+j) = (pow(i,-kappa)) * l_eps_init(1+j) + (1.0 - pow(i,-kappa)) * l_eps_bar(1+j);
          
          //update the mass matrix
          M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
          //M_disc(j) = 1.0 / M_inv_disc(j);
          
        }
        
        //updating the global and local step size once more if k == d
        if(k == d){
          l_eps_init(0) = arma::min(l_eps_init.subvec(1,k));
          l_eps_bar(0) = arma::min(l_eps_bar.subvec(1,k));
          
          //reupdate the mass matrix
          for(j = 0; j < k; j++){
            M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
            //M_disc(j) = 1.0 / M_inv_disc(j);
          }
        }
        
      }else{
        if(k == d){
          //pure dnuts case
          gain = delta(1) - arma::mean(alpha);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
        }else if(k == 0 || std::isnan(delta(1))){
          //clasic nuts case
          
          gain = delta(0) - alpha(0); 
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }else{
          //mixed dnuts case
          gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); 
          //+ arma::var(alpha.subvec(1,k));
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }
        
        //update the moving average of the statistic to be brought to zero
        Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
        
        //update the estimate of the logarithm of the step size
        l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
        
        //update the moving average of the log value of the step size
        l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
      }
      
      
    }
  }else{
    // dhmc
    
    //sequentially updates the step size
    for(unsigned int i = 1; i <= N; i++){
      
      //do a hmc iteration
      if(k == 0){
        //classic hmc
        
        //generate particle momentum
        m = M_cont * arma::randn(d);
        
        //hmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init(0)),
                                max_treedepth,
                                d,
                                M_inv_cont);
        
      }else if(k == d){
        //pure dhmc
        
        //generate particle momentum
        m = M_disc % rlaplace(d);
        
        //dhmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init(0)),
                                max_treedepth,
                                d,
                                idx_disc,
                                M_inv_disc);
      }else {
        //dhmc
        
        //generate particle momentum
        m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //dhmc iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                std::exp(l_eps_init(0)),
                                max_treedepth,
                                d,
                                k,
                                idx_disc,
                                M_inv_cont,
                                M_inv_disc);
      }
      
      //update the position value of the particle
      theta = Rcpp::as<arma::vec>(iteration["theta"]);
      
      //get alpha value and ensure it is admissible
      alpha = Rcpp::as<arma::vec>(iteration["alpha"]);
      
      //check condition for different stepsize
      if(condition){
        //different step size achieved by updating the discontinuous Mass Matrix
        
        //updating the global step size
        
        if(k != d){
          //global gain
          gain = delta(0) - alpha(0);
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
          
        }else{
          start = 0;
        }
        
        
        //updating the single discontinuous components Mass Matrix 
        for(j = 0; j < k; j++){
          
          //local gain
          gain = delta(1) - alpha(start+j);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
          
          //update the moving average of the statistic to be brought to zero
          Hbar(1+j) = (1 - 1/(t0+i))*Hbar(1+j) + gain / (t0 + i);
          
          //update the estimate of the logarithm of the step size
          l_eps_init(1+j) = mu(1+j) - Hbar(1+j) * std::sqrt(static_cast<double>(i)) / gamma;
          
          //update the moving average of the log value of the step size
          l_eps_bar(1+j) = (pow(i,-kappa)) * l_eps_init(1+j) + (1.0 - pow(i,-kappa)) * l_eps_bar(1+j);
          
          //update the mass matrix
          M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
          //M_disc(j) = 1.0 / M_inv_disc(j);
          
        }
        
        //updating the global and local step size once more if k == d
        if(k == d){
          l_eps_init(0) = arma::min(l_eps_init.subvec(1,k));
          l_eps_bar(0) = arma::min(l_eps_bar.subvec(1,k));
          
          //reupdate the mass matrix
          for(j = 0; j < k; j++){
            M_inv_disc(j) = std::exp(l_eps_init(1+j) - l_eps_init(0));
            //M_disc(j) = 1.0 / M_inv_disc(j);
          }
        }
        
      }else{
        if(k == d){
          //pure dnuts case
          gain = delta(1) - arma::mean(alpha);
          if(!arma::is_finite(gain)){
            gain = delta(1);
          }
        }else if(k == 0 || std::isnan(delta(1))){
          //clasic nuts case
          
          gain = delta(0) - alpha(0); 
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }else{
          //mixed dnuts case
          gain = delta(0) - alpha(0) + arma::mean(delta(1) - alpha.subvec(1,k)); 
          //+ arma::var(alpha.subvec(1,k));
          if(!arma::is_finite(gain)){
            gain = delta(0);
          }
        }
        
        //update the moving average of the statistic to be brought to zero
        Hbar(0) = (1 - 1/(t0+i))*Hbar(0) + gain / (t0 + i);
        
        //update the estimate of the logarithm of the step size
        l_eps_init(0) = mu(0) - Hbar(0) * std::sqrt(static_cast<double>(i)) / gamma;
        
        //update the moving average of the log value of the step size
        l_eps_bar(0) = (pow(i,-kappa)) * l_eps_init(0) + (1.0 - pow(i,-kappa)) * l_eps_bar(0);
      }

    }
  }
  
  //update position value and step size 
  theta0 = theta;
  eps0 = std::exp(l_eps_bar(0));
  //for different step size also update the discontinuous components mass matrix
  if(condition){
    for(j = 0; j < k; j++){
      M_inv_disc(j) = std::exp(l_eps_bar(1 + j) - l_eps_bar(0));
      M_disc(j) = 1.0 / M_inv_disc(j);
    }
  }
  
}

// WRAPPER FUNCTION

void adapt_stepsize_wrapper(arma::vec& theta,
                            double& eps,
                            const Rcpp::Function& nlp,
                            const Rcpp::List& args,
                            const unsigned int& d,
                            const unsigned int& k,
                            arma::uvec& idx_disc,
                            const unsigned int& N_init,
                            const Rcpp::List& control,
                            const arma::vec& M_cont_diag,
                            arma::vec& M_disc,
                            const arma::vec& M_inv_cont_diag,
                            arma::vec& M_inv_disc,
                            const arma::mat& M_cont_dense,
                            const arma::mat& M_inv_cont_dense,
                            const std::string& M_type,
                            const double& log_tau,
                            const unsigned int& L,
                            const bool& verbose,
                            const unsigned int& chain_id){
  
  //initialize momentum vector
  arma::vec m(d);
  
  //initialize the epsilon vector
  arma::vec epsilon;
  
  if(M_type == "identity" || (M_cont_diag.n_elem == 0 && M_cont_dense.n_elem == 0 && M_disc.n_elem == 0)){
    //identity matrix case
    
    //epsilon initialization?
    if(std::isnan(eps)){
      
      if(k == 0){
        //classic nuts
        
        //generate momentum
        m = arma::randn(d); //gauss
        
        //initialization
        eps = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           0);
        
      }else if(k == d){
        //pure dnuts
        
        //generate momentum
        m = rlaplace(d); //laplace
        
        //initialization
        epsilon = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           idx_disc);
        
        //if different step size is enable change the inverse mass matrix
        if(Rcpp::as<bool>(control["different_stepsize"])){
          //take the mean value
          eps = arma::mean(epsilon);
          
          //update the mass matrix
          M_inv_disc = epsilon / eps;
            
        }else{
          //take the mean of epsilon as the global initial step size
          eps = arma::mean(epsilon);
        }
        
        
      }else {
        //mixed dnuts
        
        //generate momentum
        m.subvec(0,d-k-1) = arma::randn(d-k); //gauss
        m.subvec(d-k,d-1) = rlaplace(k); //laplace
        
        //initialization
        epsilon = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           k,
                           idx_disc);
        
        //if different step size is enable change the inverse mass matrix
        if(Rcpp::as<bool>(control["different_stepsize"])){
          //take the global initial step size
          eps = epsilon(0);
          
          //update the discontinuous mass matrix
          M_inv_disc = epsilon.subvec(1,k) / eps;
          M_disc = arma::ones<arma::vec>(k);
          
        }else{
          //take the mean value
          eps = arma::mean(epsilon);
        }
      }
    }
    
    
    //if verbose, print the current step size
    if(verbose){
      Rcpp::Rcout << "Chain " << chain_id << ", initial step-size: " << eps << std::endl;
    }
    
    //adaptive calibration via dual averaging
    if(Rcpp::as<bool>(control["different_stepsize"])){
      //create a fictitiuous M_cont_diag matrix
      arma::vec M_cont_diag_one = arma::ones<arma::vec>(d-k);
      adapt_stepsize(theta,
                     nlp,
                     args,
                     eps,
                     d,
                     k,
                     idx_disc,
                     control,
                     N_init,
                     log_tau,
                     L,
                     M_cont_diag_one,
                     M_disc,
                     M_cont_diag_one,
                     M_inv_disc);
    }else{
      //no matrix needed
      adapt_stepsize(theta,
                     nlp,
                     args,
                     eps,
                     d,
                     k,
                     idx_disc,
                     control,
                     N_init,
                     log_tau,
                     L);
    }
      
  }else if(M_type == "diagonal"){
    //diagonal matrix case
    
    //epsilon initialization?
    if(std::isnan(eps)){
      
      if(k == 0){
        //classic nuts
        
        //generate momentum
        m = M_cont_diag % arma::randn(d); //gauss
        
        //initialization
        eps = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           0,
                           M_inv_cont_diag);
        
      }else if(k == d){
        //pure dnuts
        
        //generate momentum
        m = M_disc % rlaplace(d); //laplace
        
        //initialization
        epsilon = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           idx_disc,
                           M_inv_disc);
        
        //if different step size is enable change the inverse mass matrix
        if(Rcpp::as<bool>(control["different_stepsize"])){
          //take the mean value
          eps = arma::mean(epsilon);
          
          //update the mass matrix
          M_inv_disc %= epsilon / eps;
          
        }else{
          //take the mean of epsilon as the global initial step size
          eps = arma::mean(epsilon);
        }
        
      }else {
        //mixed dnuts
        
        //generate momentum
        m.subvec(0,d-k-1) = M_cont_diag % arma::randn(d-k); //gauss
        m.subvec(d-k,d-1) = M_disc % rlaplace(k); //laplace
        
        //initialization
        epsilon = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           k,
                           idx_disc,
                           M_inv_cont_diag,
                           M_inv_disc);
        
        //if different step size is enable change the inverse mass matrix
        if(Rcpp::as<bool>(control["different_stepsize"])){
          //take the first value
          eps = epsilon(0);
          
          //update the mass matrix
          M_inv_disc %= epsilon.subvec(1,k) / eps;
          
        }else{
          //take the mean of epsilon as the global initial step size
          eps = arma::mean(epsilon);
        }
      }
    }
    
    //adaptive calibration via dual averaging
    adapt_stepsize(theta,
                   nlp,
                   args,
                   eps,
                   d,
                   k,
                   idx_disc,
                   control,
                   N_init,
                   log_tau,
                   L,
                   M_cont_diag,
                   M_disc,
                   M_inv_cont_diag,
                   M_inv_disc);
      
  }else if(M_type == "dense"){
    //dense matrix case
    
    //epsilon initialization?
    if(std::isnan(eps)){
      
      if(k == 0){
        //classic nuts
        
        //generate momentum
        m = M_cont_dense * arma::randn(d); //gauss
        
        //initialization
        eps = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           0,
                           M_inv_cont_dense);
        
      }else if(k == d){
        //pure dnuts
        
        //generate momentum
        m = M_disc % rlaplace(d); //laplace
        
        //initialization
        epsilon = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           idx_disc,
                           M_inv_disc);
        
        //if different step size is enable change the inverse mass matrix
        if(Rcpp::as<bool>(control["different_stepsize"])){
          //take the mean value
          eps = arma::mean(epsilon);
          
          //update the mass matrix
          M_inv_disc %= epsilon / eps;
          
        }else{
          //take the mean of epsilon as the global initial step size
          eps = arma::mean(epsilon);
        }
        
      }else {
        //mixed dnuts
        
        //generate momentum
        m.subvec(0,d-k-1) = M_cont_dense * arma::randn(d-k); //gauss
        m.subvec(d-k,d-1) = M_disc % rlaplace(k); //laplace
        
        //initialization
        epsilon = init_epsilon(theta,
                           m,
                           nlp,
                           args,
                           d,
                           k,
                           idx_disc,
                           M_inv_cont_dense,
                           M_inv_disc);
        
        //if different step size is enable change the inverse mass matrix
        if(Rcpp::as<bool>(control["different_stepsize"])){
          //take the first value
          eps = epsilon(0);
          
          //update the mass matrix
          M_inv_disc %= epsilon.subvec(1,k) / eps;
          
        }else{
          //take the mean of epsilon as the global initial step size
          eps = arma::mean(epsilon);
        }
      }
    }
    
    //adaptive calibration via dual averaging
    adapt_stepsize(theta,
                   nlp,
                   args,
                   eps,
                   d,
                   k,
                   idx_disc,
                   control,
                   N_init,
                   log_tau,
                   L,
                   M_cont_dense,
                   M_disc,
                   M_inv_cont_dense,
                   M_inv_disc);
    
  }
}
