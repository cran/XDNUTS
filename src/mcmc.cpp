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

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// FUNCTIONS THAT DO MCMC

// caso matrice identità senza riciclo
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
          const unsigned int& L_jitter){
  
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx = sequenza(N,refresh);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize alpha to zero
  alpha *= 0.0;
  
  //list containing the output of an iteration
  Rcpp::List iteration;
  
  //current position of the particle
  arma::vec theta = theta0;
  
  //current momentum of the particle
  arma::vec m = arma::zeros<arma::vec>(d);
  
  //initialize j, index for the thinned samples
  int j;
  
  //calculate the current value of energy with momentum equal to zero
  double H0 = Rcpp::as<double>(nlp(theta0,args,1));
  
  //define the energy leap
  double delta_E = 0;
  
  //let's distinguish the various algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = arma::randn(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = rlaplace(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   idx_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else{
        // dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = arma::randn(d-k);
          m.subvec(d-k,d-1) = rlaplace(k);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   k,
                                   idx_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = H0;
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else if(L == 0){
    //xdhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = arma::randn(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   log_tau);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = rlaplace(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   idx_disc,
                                   std::exp(log_tau));
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else{
        // dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = arma::randn(d-k);
          m.subvec(d-k,d-1) = rlaplace(k);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   k,
                                   idx_disc,
                                   log_tau);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = H0;
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else{
    //dhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = arma::randn(d);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = rlaplace(d);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  idx_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else{
        // dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = arma::randn(d-k);
          m.subvec(d-k,d-1) = rlaplace(k);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  k,
                                  idx_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = H0;
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }
  
  if(warm_up){
    Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
  }else{
    Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
  }   
  //update current value
  theta0 = theta;
}



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
          const arma::vec& M_inv_disc){
  
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx = sequenza(N,refresh);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize alpha to zero
  alpha *= 0.0;
  
  //list containing the output of an iteration
  Rcpp::List iteration;
  
  //current position of the particle
  arma::vec theta = theta0;
  
  //current momentum of the particle
  arma::vec m = arma::zeros<arma::vec>(d);
  
  //initialize j, index for the thinned samples
  int j;
  
  //calculate the current value of energy with momentum equal to zero
  double H0 = Rcpp::as<double>(nlp(theta0,args,1));
  
  //define the energy leap
  double delta_E = 0;
  
  //let's distinguish the various algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_cont % arma::randn(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   M_inv_cont);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_disc % rlaplace(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   idx_disc,
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else{
        // dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
          m.subvec(d-k,d-1) = M_disc % rlaplace(k);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   k,
                                   idx_disc,
                                   M_inv_cont,
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else if(L == 0){
    // xdhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_cont % arma::randn(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   log_tau,
                                   M_inv_cont);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_disc % rlaplace(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   idx_disc,
                                   std::exp(log_tau),
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else{
        // dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
          m.subvec(d-k,d-1) = M_disc % rlaplace(k);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   k,
                                   idx_disc,
                                   log_tau,
                                   M_inv_cont,
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else{
    // dhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_cont % arma::randn(d);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  M_inv_cont);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_disc % rlaplace(d);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  idx_disc,
                                  M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else{
        // dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
          m.subvec(d-k,d-1) = M_disc % rlaplace(k);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  k,
                                  idx_disc,
                                  M_inv_cont,
                                  M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }
  
  
  if(warm_up){
    Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
  }else{
    Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
  }   
  //update current value
  theta0 = theta;  
}


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
          const arma::vec& M_inv_disc){
  
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx = sequenza(N,refresh);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize alpha to zero
  alpha *= 0.0;
  
  //list containing the output of an iteration
  Rcpp::List iteration;
  
  //current position of the particle
  arma::vec theta = theta0;
  
  //current momentum of the particle
  arma::vec m = arma::zeros<arma::vec>(d);
  
  //inizializza il valore di j
  int j;
  
  //calculate the current value of energy with momentum equal to zero
  double H0 = Rcpp::as<double>(nlp(theta0,args,1));
  
  //define the energy leap
  double delta_E = 0;
  
  //let's distinguish the various algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts 
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_cont * arma::randn(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   M_inv_cont);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_disc % rlaplace(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   idx_disc,
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
        
      }else {
        // dnuts
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
          m.subvec(d-k,d-1) = M_disc % rlaplace(k);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   k,
                                   idx_disc,
                                   M_inv_cont,
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
    
  }else if(L == 0){
    // xdhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_cont * arma::randn(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   log_tau,
                                   M_inv_cont);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_disc % rlaplace(d);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   idx_disc,
                                   std::exp(log_tau),
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
        
      }else {
        // dnuts
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
          m.subvec(d-k,d-1) = M_disc % rlaplace(k);
          
          //do a iteration
          iteration = nuts_singolo(theta,
                                   m,
                                   nlp,
                                   args,
                                   step_size(i),
                                   max_treedepth,
                                   d,
                                   k,
                                   idx_disc,
                                   log_tau,
                                   M_inv_cont,
                                   M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else{
    // dhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_cont * arma::randn(d);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  M_inv_cont);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }else if(k == d){
        //pure dnuts
        
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m = M_disc % rlaplace(d);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  idx_disc,
                                  M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
        
      }else {
        // dnuts
        for(j = 0; j < thin; j++){
          //generate the momentum of the particle
          m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
          m.subvec(d-k,d-1) = M_disc % rlaplace(k);
          
          //do a iteration
          iteration = hmc_singolo(theta,
                                  m,
                                  nlp,
                                  args,
                                  step_size(i),
                                  sample_step_length(L,L_jitter),
                                  d,
                                  k,
                                  idx_disc,
                                  M_inv_cont,
                                  M_inv_disc);
          
          theta = Rcpp::as<arma::vec>(iteration["theta"]);
          //if you get to the last iteration calculate the delta E
          if(j == (thin - 1)){
            delta_E = Rcpp::as<double>(iteration["E"]) - H0;
          }
          
          //update the energy value
          H0 = Rcpp::as<double>(iteration["E"]);
        }
        
      }
      
      
      
      //mcmc
      out.row(i) = theta.t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap done
        delta_energy(i) = delta_E;
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }
  
  if(warm_up){
    Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
  }else{
    Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
  }   
  //update current value
  theta0 = theta;
}

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
          const unsigned int& K){
  
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx = sequenza(N,refresh);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize alpha to zero
  alpha *= 0.0;
  
  //list containing the output of an iteration
  Rcpp::List iteration;
  
  //current position of the particle
  arma::vec theta = theta0;
  
  //current momentum of the particle
  arma::vec m = arma::zeros<arma::vec>(d);
  
  //calculate the current value of energy with momentum equal to zero
  double H0 = Rcpp::as<double>(nlp(theta0,args,1));
  
  //let's distinguish the various algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = arma::randn(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = rlaplace(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = arma::randn(d-k);
        m.subvec(d-k,d-1) = rlaplace(k);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 K);
      }
      
      
      //mcmc

      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();

      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else if(L == 0){
    //xdhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = arma::randn(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 log_tau,
                                 K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = rlaplace(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 std::exp(log_tau),
                                 K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = arma::randn(d-k);
        m.subvec(d-k,d-1) = rlaplace(k);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 log_tau,
                                 K);
      }
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else{
    // dhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = arma::randn(d);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = rlaplace(d);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                idx_disc,
                                K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = arma::randn(d-k);
        m.subvec(d-k,d-1) = rlaplace(k);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                k,
                                idx_disc,
                                K);
      }
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }
  if(warm_up){
    Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
  }else{
    Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
  }   
  //update current value
  theta0 = theta;  
}

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
          const unsigned int& K){
  
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx = sequenza(N,refresh);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize alpha to zero
  alpha *= 0.0;
  
  //list containing the output of an iteration
  Rcpp::List iteration;
  
  //current position of the particle
  arma::vec theta = theta0;
  
  //current momentum of the particle
  arma::vec m = arma::zeros<arma::vec>(d);
  
  //calculate the current value of energy with momentum equal to zero
  double H0 = Rcpp::as<double>(nlp(theta0,args,1));
  
  //let's distinguish the various algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = M_cont % arma::randn(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 M_inv_cont,
                                 K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = M_disc % rlaplace(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 M_inv_disc,
                                 K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 M_inv_cont,
                                 M_inv_disc,
                                 K);
      }
      
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else if(L == 0){
    //xdhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = M_cont % arma::randn(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 log_tau,
                                 M_inv_cont,
                                 K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = M_disc % rlaplace(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 std::exp(log_tau),
                                 M_inv_disc,
                                 K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 log_tau,
                                 M_inv_cont,
                                 M_inv_disc,
                                 K);
      }
      
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else {
    // dhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = M_cont % arma::randn(d);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                M_inv_cont,
                                K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = M_disc % rlaplace(d);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                idx_disc,
                                M_inv_disc,
                                K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = M_cont % arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                k,
                                idx_disc,
                                M_inv_cont,
                                M_inv_disc,
                                K);
      }
      
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }
  if(warm_up){
    Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
  }else{
    Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
  }   
  //update current value
  theta0 = theta;
}

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
          const unsigned int& K){
  
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx = sequenza(N,refresh);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize alpha to zero
  alpha *= 0.0;
  
  //list containing the output of an iteration
  Rcpp::List iteration;
  
  //current position of the particle
  arma::vec theta = theta0;
  
  //current momentum of the particle
  arma::vec m = arma::zeros<arma::vec>(d);
  
  //calculate the current value of energy with momentum equal to zero
  double H0 = Rcpp::as<double>(nlp(theta0,args,1));
  
  //let's distinguish the various algorithms
  if(L == 0 && log_tau == 1000){
    //dnuts
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = M_cont * arma::randn(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 M_inv_cont,
                                 K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = M_disc % rlaplace(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 M_inv_disc,
                                 K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 M_inv_cont,
                                 M_inv_disc,
                                 K);
      }
      
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
  }else if(L == 0){
    //xdhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = M_cont * arma::randn(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 log_tau,
                                 M_inv_cont,
                                 K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = M_disc % rlaplace(d);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 idx_disc,
                                 std::exp(log_tau),
                                 M_inv_disc,
                                 K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //do a iteration
        iteration = nuts_singolo(theta,
                                 m,
                                 nlp,
                                 args,
                                 step_size(i),
                                 max_treedepth,
                                 d,
                                 k,
                                 idx_disc,
                                 log_tau,
                                 M_inv_cont,
                                 M_inv_disc,
                                 K);
      }
      
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
        
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
    
  }else {
    // dhmc
    
    //for loop
    for(unsigned int i = 0; i < N; i++){
      
      if(k == 0){
        //classi nuts
        
        //generate the momentum of the particle
        m = M_cont * arma::randn(d);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                M_inv_cont,
                                K);
        
      }else if(k == d){
        //pure dnuts
        
        //generate the momentum of the particle
        m = M_disc % rlaplace(d);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                idx_disc,
                                M_inv_disc,
                                K);
        
      }else {
        // dnuts
        
        //generate the momentum of the particle
        m.subvec(0,d-k-1) = M_cont * arma::randn(d-k);
        m.subvec(d-k,d-1) = M_disc % rlaplace(k);
        
        //do a iteration
        iteration = hmc_singolo(theta,
                                m,
                                nlp,
                                args,
                                step_size(i),
                                sample_step_length(L,L_jitter),
                                d,
                                k,
                                idx_disc,
                                M_inv_cont,
                                M_inv_disc,
                                K);
      }
      
      
      
      //mcmc
      
      //recycled samples
      out.rows(K*i,K*i + K - 1) = arma::reshape(Rcpp::as<arma::vec>(iteration["theta"]),d,K).t();
      
      //update the current value of the chain
      theta = out.row(K*i).t();
      
      //check if the console update condition is met
      if(i == seq_idx(conta)){
        //print to console
        if(warm_up){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }else{
          Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta+1) * 1000) / 10.0 << "%" << std::endl;
        }      
        //update the count
        conta++;
      }
      
      //if we are in the sampling phase we also save other quantities of interest
      if(!warm_up){
        //length of the trajectory (numbers of integrations made)
        step_length(i) = Rcpp::as<double>(iteration["n"]);
        
        //explored energy level
        energy(i) = Rcpp::as<double>(iteration["E"]);
        
        //energy leap
        delta_energy(i) = energy(i) - H0;
        H0 = energy(i);
        
        //estimated average acceptance rates
        alpha += Rcpp::as<arma::vec>(iteration["alpha"]);
      }
    }
    
  }
  if(warm_up){
    Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
  }else{
    Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
  }   
  //update current value
  theta0 = theta;  
}

//  wrapper function
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
                  const unsigned int& L_jitter){
  
  if(M_type == "identity" || (M_cont_diag.n_elem == 0 && M_cont_dense.n_elem == 0 && M_disc.n_elem == 0)){
    //identity case
    
    if(K == 1 || (M_type == "identity" && warm_up)){
      //without recycling samples
      
      mcmc(theta,
           nlp,
           args,
           N,
           d,
           k,
           idx_disc,
           max_treedepth,
           refresh,
           out,
           step_size,
           step_length,
           energy,
           delta_energy,
           alpha,
           warm_up,
           chain_id,
           thin,
           log_tau,
           L,
           L_jitter);
      
    }else{
      //with recycling samples
      mcmc(theta,
           nlp,
           args,
           N,
           d,
           k,
           idx_disc,
           max_treedepth,
           refresh,
           out,
           step_size,
           step_length,
           energy,
           delta_energy,
           alpha,
           warm_up,
           chain_id,
           log_tau,
           L,
           L_jitter,
           K);
    }
    
    // if we don't want the identity but we are at the beginning, we update the estimates
    if(warm_up && M_type != "identity"){
      // mass matrix estimation
      update_MM(M_cont_diag,
                M_disc,
                M_inv_cont_diag,
                M_inv_disc,
                M_cont_dense,
                M_inv_cont_dense,
                out,
                N,
                K,
                bar,
                d,
                k,
                M_type);
    }
    
  }else{
    
    if(M_type == "diagonal"){
      //diagonal case
      if(K == 1){
        //without recycling
        mcmc(theta,
             nlp,
             args,
             N,
             d,
             k,
             idx_disc,
             max_treedepth,
             refresh,
             out,
             step_size,
             step_length,
             energy,
             delta_energy,
             alpha,
             warm_up,
             chain_id,
             thin,
             log_tau,
             L,
             L_jitter,
             M_cont_diag,
             M_disc,
             M_inv_cont_diag,
             M_inv_disc);
        
      }else{
        //with recycling
        mcmc(theta,
             nlp,
             args,
             N,
             d,
             k,
             idx_disc,
             max_treedepth,
             refresh,
             out,
             step_size,
             step_length,
             energy,
             delta_energy,
             alpha,
             warm_up,
             chain_id,
             log_tau,
             L,
             L_jitter,
             M_cont_diag,
             M_disc,
             M_inv_cont_diag,
             M_inv_disc,
             K);
      }
      
    }else if(M_type == "dense"){
      //dense case
      if(K == 1){
        //without recycling
        mcmc(theta,
             nlp,
             args,
             N,
             d,
             k,
             idx_disc,
             max_treedepth,
             refresh,
             out,
             step_size,
             step_length,
             energy,
             delta_energy,
             alpha,
             warm_up,
             chain_id,
             thin,
             log_tau,
             L,
             L_jitter,
             M_cont_dense,
             M_disc,
             M_inv_cont_dense,
             M_inv_disc);
        
      }else{
        //with recycling
        mcmc(theta,
             nlp,
             args,
             N,
             d,
             k,
             idx_disc,
             max_treedepth,
             refresh,
             out,
             step_size,
             step_length,
             energy,
             delta_energy,
             alpha,
             warm_up,
             chain_id,
             log_tau,
             L,
             L_jitter,
             M_cont_dense,
             M_disc,
             M_inv_cont_dense,
             M_inv_disc,
             K);
      }
    }
    
    if(warm_up){
      // mass matrix estimation
      update_MM(M_cont_diag,
                M_disc,
                M_inv_cont_diag,
                M_inv_disc,
                M_cont_dense,
                M_inv_cont_dense,
                out,
                N,
                K,
                bar,
                d,
                k,
                M_type);
    }
  }
  
}
