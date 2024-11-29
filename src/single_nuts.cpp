#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "recursive_tree.h"
#include "single_nuts.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

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
                        arma::uvec& idx_disc){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+4+k);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-k-1))) + 
    arma::sum(arma::abs(sub_tree.subvec(2*d-k,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(6*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-k-1) = sub_tree.subvec(d,2*d-k-1); //gauss
  sub_tree.subvec(6*d-k,6*d-1) = arma::sign(sub_tree.subvec(2*d-k,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+4+k);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2(6*d+1)){
      if(arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //and update the logarithm of the weights for multinomial sampling from the trajectory
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,k);
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(6*d+1,6*d+3+k) += sub_tree2.subvec(6*d+1,6*d+3+k);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+2,6*d+2+k) / sub_tree(6*d+3+k),
                            Rcpp::Named("n") = sub_tree(6*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const arma::vec& M_inv_disc){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+4+k);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-k-1)),M_inv_cont ) + 
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(6*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-k-1) = sub_tree.subvec(d,2*d-k-1); //gauss
  sub_tree.subvec(6*d-k,6*d-1) = arma::sign(sub_tree.subvec(2*d-k,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+4+k);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2(6*d+1)){
      if(arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //and update the logarithm of the weights for multinomial sampling from the trajectory
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,k,M_inv_cont,M_inv_disc);
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(6*d+1,6*d+3+k) += sub_tree2.subvec(6*d+1,6*d+3+k);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+2,6*d+2+k) / sub_tree(6*d+3+k),
                            Rcpp::Named("n") = sub_tree(6*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const arma::vec& M_inv_disc){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+4+k);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-k-1), M_inv_cont * sub_tree.subvec(d,2*d-k-1)) +
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(6*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-k-1) = sub_tree.subvec(d,2*d-k-1); //gauss
  sub_tree.subvec(6*d-k,6*d-1) = arma::sign(sub_tree.subvec(2*d-k,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+4+k);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2(6*d+1)){
      if(arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //and update the logarithm of the weights for multinomial sampling from the trajectory
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,k,M_inv_cont,M_inv_disc);
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(6*d+1,6*d+3+k) += sub_tree2.subvec(6*d+1,6*d+3+k);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+2,6*d+2+k) / sub_tree(6*d+3+k),
                            Rcpp::Named("n") = sub_tree(6*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+4+k);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-k-1))) + 
    arma::sum(arma::abs(sub_tree.subvec(2*d-k,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((5+K)*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-k-1) = sub_tree.subvec(d,2*d-k-1); //gauss
  sub_tree.subvec((5+K)*d-k,(5+K)*d-1) = arma::sign(sub_tree.subvec(2*d-k,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+4+k);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2((5+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      // 
      //uniform for the others
      
      //sample from the trajectory
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      
      //cumulate the log multinomial weights
      sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
      
      //recalculate alpha
      //alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,k,K);
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((5+K)*d+1,(5+K)*d+3+k) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3+k);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+2,(5+K)*d+2+k) / sub_tree((5+K)*d+3+k),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+4+k);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-k-1)),M_inv_cont ) +
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((5+K)*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-k-1) = sub_tree.subvec(d,2*d-k-1); //gauss
  sub_tree.subvec((5+K)*d-k,(5+K)*d-1) = arma::sign(sub_tree.subvec(2*d-k,2*d-1)); //laplace

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+4+k);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc,
                             K);

      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2((5+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      
      //sample from the trajectory
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      
      //uniform for the others
      
      //cumulate the log multinomial weights
      sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
      
      //recalculate alpha
      //alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,k,M_inv_cont,M_inv_disc,K);
      
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((5+K)*d+1,(5+K)*d+3+k) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3+k);
    
    //increase the depth of the tree
    depth++;
    
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+2,(5+K)*d+2+k) / sub_tree((5+K)*d+3+k),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+4+k);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-k-1), M_inv_cont * sub_tree.subvec(d,2*d-k-1)) + 
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((5+K)*d) = -H0;
  
  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-k-1) = sub_tree.subvec(d,2*d-k-1); //gauss
  sub_tree.subvec((5+K)*d-k,(5+K)*d-1) = arma::sign(sub_tree.subvec(2*d-k,2*d-1)); //laplace

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+4+k);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }

    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2((5+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      
      //uniform for the others
      
      //sample from the trajectory
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      
      //cumulate the log multinomial weights
      sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
      
      //recalculate alpha
      //alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,k,M_inv_cont,M_inv_disc,K);
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((5+K)*d+1,(5+K)*d+3+k) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3+k);
    
    //increase the depth of the tree
    depth++;

  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+2,(5+K)*d+2+k) / sub_tree((5+K)*d+3+k),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3+k),
                            Rcpp::Named("E") = H0);
}

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
                        const unsigned int& d){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;

  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;

  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;

  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;

  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(6*d) = -H0;
  
  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-1) = sub_tree.subvec(d,2*d-1); //gauss

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+4);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d);

      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2(6*d+1)){
      
      if(arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
        
      }
      
      //and update the logarithm of the weights for multinomial sampling from the trajectory
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d);
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(6*d+1,6*d+3) += sub_tree2.subvec(6*d+1,6*d+3);
    
    //increase the depth of the tree
    depth++;
    
    
    
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+2,6*d+2) / sub_tree(6*d+3),
                            Rcpp::Named("n") = sub_tree(6*d+3),
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::vec& M_inv){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-1)),M_inv);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(6*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-1) = sub_tree.subvec(d,2*d-1); //gauss
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+4);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             M_inv);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             M_inv);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2(6*d+1)){
      if(arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //and update the logarithm of the weights for multinomial sampling from the trajectory
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,M_inv);
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(6*d+1,6*d+3) += sub_tree2.subvec(6*d+1,6*d+3);

    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+2,6*d+2) / sub_tree(6*d+3),
                            Rcpp::Named("n") = sub_tree(6*d+3),
                            Rcpp::Named("E") = H0);
}

// dense matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::mat& M_inv){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-1),M_inv * sub_tree.subvec(d,2*d-1));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(6*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-1) = sub_tree.subvec(d,2*d-1); //gauss
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+4);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             M_inv);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             M_inv);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2(6*d+1)){
      if(arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //and update the logarithm of the weights for multinomial sampling from the trajectory
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,M_inv);
      
    }
    
    //cumula il resto:tassi di accettazione metropolis e numero di foglie
    sub_tree.subvec(6*d+1,6*d+3) += sub_tree2.subvec(6*d+1,6*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+2,6*d+2) / sub_tree(6*d+3),
                            Rcpp::Named("n") = sub_tree(6*d+3),
                            Rcpp::Named("E") = H0);
}

/* ---------------------------- RECYCLED VERSION ---------------------------- */


// identity matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d, 
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-1)));
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((5+K)*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-1) = sub_tree.subvec(d,2*d-1); //gauss
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+4);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2((5+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      
      //sample from the trajectory
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      
      //uniform for the others
      
      //cumulate the log multinomial weights
      sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
      
      //recalculate alpha
      //alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,K);
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((5+K)*d+1,(5+K)*d+3) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+2,(5+K)*d+2) / sub_tree((5+K)*d+3),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3),
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::vec& M_inv,
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-1)),M_inv);
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((5+K)*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-1) = sub_tree.subvec(d,2*d-1); //gauss
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+4);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             M_inv,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             M_inv,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2((5+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      
      //sample from the trajectory
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      
      //uniform for the others
      
      //cumulate the log multinomial weights
      sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
      
      //recalculate alpha
      //alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,M_inv,K);
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((5+K)*d+1,(5+K)*d+3) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+2,(5+K)*d+2) / sub_tree((5+K)*d+3),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3),
                            Rcpp::Named("E") = H0);
}

// dense matrix case with recycle
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const arma::mat& M_inv,
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-1),M_inv * sub_tree.subvec(d,2*d-1));
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((5+K)*d) = -H0;

  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-1) = sub_tree.subvec(d,2*d-1); //gauss
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+4);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             M_inv,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             M_inv,
                             K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree2((5+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      
      //sample from the trajectory
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      
      //uniform for the others
      
      //cumulate the log multinomial weights
      sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
      
      //recalculate alpha
      //alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,M_inv,K);
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((5+K)*d+1,(5+K)*d+3) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+2,(5+K)*d+2) / sub_tree((5+K)*d+3),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3),
                            Rcpp::Named("E") = H0);
}

/* -------------------------------------------------------------------------- */

/* ------------------ VERSION WITH ONLY DISCRETE PARAMETERS ----------------- */

/* -------------------------------------------------------------------------- */


// identity matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                         const arma::vec& m0,
                         const Rcpp::Function& nlp,
                         const Rcpp::List& args,
                         const double& eps,
                         const unsigned int& max_treedepth,
                         const unsigned int& d,
                         arma::uvec& idx_disc){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(7*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree(6*d) = sub_tree(6*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree(6*d) + arma::sum(arma::abs(m0));
  
  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-1) = arma::sign(sub_tree.subvec(d,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(7*d+4);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree(6*d + 2) = sub_tree2(6*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree(6*d) = sub_tree2(6*d);
      }
      
    }
    
    //if haven't encountered any divergences, update the trajectory proposed value
    if(!sub_tree2(6*d+1)){
      //if(arma::randu() < 0.5){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      //}
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn2(sub_tree,d);
      
    }
    
    //cumulate the reminder: metropolis rate and number of leaves
    sub_tree(6*d+1) += sub_tree2(6*d+1);
    sub_tree.subvec(6*d+3,7*d+3) += sub_tree2.subvec(6*d+3,7*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+3,7*d+2) / sub_tree(7*d+3),
                            Rcpp::Named("n") = sub_tree(7*d+3),
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                         const arma::vec& m0,
                         const Rcpp::Function& nlp,
                         const Rcpp::List& args,
                         const double& eps,
                         const unsigned int& max_treedepth,
                         const unsigned int& d,
                         arma::uvec& idx_disc,
                         const arma::vec& M_inv){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(7*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree(6*d) = sub_tree(6*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree(6*d) + arma::sum(M_inv % arma::abs(m0));
  
  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec(5*d,6*d-1) = arma::sign(sub_tree.subvec(d,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(7*d+4);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(6*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              M_inv);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree(6*d + 2) = sub_tree2(6*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              M_inv);
      
      if(!sub_tree2(6*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree(6*d) = sub_tree2(6*d);
      }
      
    }
    
    //if haven't encountered any divergences, update the trajectory proposed value
    if(!sub_tree2(6*d+1)){
      
      //if(arma::randu() < 0.5){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      //}
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn2(sub_tree,d,M_inv);
    }
    
    //cumulate the reminder: metropolis rate and number of leaves
    sub_tree(6*d+1) += sub_tree2(6*d+1);
    sub_tree.subvec(6*d+3,7*d+3) += sub_tree2.subvec(6*d+3,7*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(6*d+3,7*d+2) / sub_tree(7*d+3),
                            Rcpp::Named("n") = sub_tree(7*d+3),
                            Rcpp::Named("E") = H0);
}


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
                         const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((6+K)*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree((5+K)*d) = sub_tree((5+K)*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree((5+K)*d) + arma::sum(arma::abs(m0));
  
  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-1) = arma::sign(sub_tree.subvec(d,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((6+K)*d+5);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree((5+K)*d + 2) = sub_tree2((5+K)*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree((5+K)*d) = sub_tree2((5+K)*d);
      }
      
    }
    
    //if haven't encountered any divergences, update the trajectory extreme value
    //and those proposed with uniform unbiased probability
    if(!sub_tree2((5+K)*d+1)){
      
      //biased probability for the proposed value
      //sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      sub_tree.subvec(4*d,(4+K)*d-1) = sub_tree2.subvec(4*d,(4+K)*d-1);
      
      //uniform for the others
      // for(unsigned int i = 1; i < K; i++){
      //   if(arma::randu() < 0.5){
      //     sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
      //   }
      // }
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec2(sub_tree,d,K);
    }
    
    //cumulate the reminder: metropolis rate and number of leaves
    sub_tree((5+K)*d+1) += sub_tree2((5+K)*d+1);
    sub_tree.subvec((5+K)*d+3,(6+K)*d+3) += sub_tree2.subvec((5+K)*d+3,(6+K)*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+3,(6+K)*d+2) / sub_tree((6+K)*d+3),
                            Rcpp::Named("n") = sub_tree((6+K)*d+3),
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                         const arma::vec& m0,
                         const Rcpp::Function& nlp,
                         const Rcpp::List& args,
                         const double& eps,
                         const unsigned int& max_treedepth,
                         const unsigned int& d,
                         arma::uvec& idx_disc,
                         const arma::vec& M_inv,
                         const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((6+K)*d+4);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree((5+K)*d) = sub_tree((5+K)*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree((5+K)*d) + arma::sum(M_inv % arma::abs(m0));
  
  //initialize the accumulation of gradients for kinetic energy
  sub_tree.subvec((4+K)*d,(5+K)*d-1) = arma::sign(sub_tree.subvec(d,2*d-1)); //laplace
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((6+K)*d+5);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((5+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              M_inv,
                              K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree((5+K)*d + 2) = sub_tree2((5+K)*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              M_inv,
                              K);
      
      if(!sub_tree2((5+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree((5+K)*d) = sub_tree2((5+K)*d);
      }
      
    }
    
    //if haven't encountered any divergences, update the trajectory extreme value
    //and those proposed with uniform unbiased probability
    if(!sub_tree2((5+K)*d+1)){
      
      //biased probability for the proposed value
      //sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      sub_tree.subvec(4*d,(4+K)*d-1) = sub_tree2.subvec(4*d,(4+K)*d-1);
      
      //uniform for the others
      // for(unsigned int i = 1; i < K; i++){
      //   if(arma::randu() < 0.5){
      //     sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
      //   }
      // }
      
      //and update the cumulative sum of the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec2(sub_tree,d,M_inv,K);
    }
    
    //cumulate the reminder: metropolis rate and number of leaves
    sub_tree((5+K)*d+1) += sub_tree2((5+K)*d+1);
    sub_tree.subvec((5+K)*d+3,(6+K)*d+3) += sub_tree2.subvec((5+K)*d+3,(6+K)*d+3);
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((5+K)*d+3,(6+K)*d+2) / sub_tree((6+K)*d+3),
                            Rcpp::Named("n") = sub_tree((6+K)*d+3),
                            Rcpp::Named("E") = H0);
}
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
                        const double& log_tau){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(5*d+k+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-k-1))) + 
    arma::sum(arma::abs(sub_tree.subvec(2*d-k,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(5*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(5*d+k+6);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(5*d+1,5*d+3+k) += sub_tree2.subvec(5*d+1,5*d+3+k);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree(5*d+1)){
      if(arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //check the condition of the virial:
      
      //first, cumulate the log sum of the metropolis weights
      sub_tree(5*d) = arma::log_add_exp(sub_tree(5*d),sub_tree2(5*d));
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( 5*d+k+4),
                           sub_tree( 5*d+k+5),
                           sub_tree2(5*d+k+4),
                           sub_tree2(5*d+k+5));
      
      //next, check the termination condition       
      sub_tree(5*d+1) += 
        (sub_tree(5*d+k+4) - sub_tree(5*d) - log(1+sub_tree(5*d+k+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+2,5*d+2+k) / sub_tree(5*d+3+k),
                            Rcpp::Named("n") = sub_tree(5*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const arma::vec& M_inv_disc){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(5*d+k+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-k-1)),M_inv_cont ) + 
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(5*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(5*d+k+6);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc);
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(5*d+1,5*d+3+k) += sub_tree2.subvec(5*d+1,5*d+3+k);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree(5*d+1)){
      if(arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //check the condition of the virial:
      
      //first, cumulate the log sum of the metropolis weights
      sub_tree(5*d) = arma::log_add_exp(sub_tree(5*d),sub_tree2(5*d));
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( 5*d+k+4),
                           sub_tree( 5*d+k+5),
                           sub_tree2(5*d+k+4),
                           sub_tree2(5*d+k+5));
      
      //next, check the termination condition       
      sub_tree(5*d+1) += 
        (sub_tree(5*d+k+4) - sub_tree(5*d) - log(1+sub_tree(5*d+k+3)) ) < log_tau;
      
    }
     
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+2,5*d+2+k) / sub_tree(5*d+3+k),
                            Rcpp::Named("n") = sub_tree(5*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const arma::vec& M_inv_disc){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(5*d+k+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-k-1), M_inv_cont * sub_tree.subvec(d,2*d-k-1)) +
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(5*d) = -H0;
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(5*d+k+6);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(5*d+1,5*d+3+k) += sub_tree2.subvec(5*d+1,5*d+3+k);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree(5*d+1)){
      if(arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //check the condition of the virial:
      
      //first, cumulate the log sum of the metropolis weights
      sub_tree(5*d) = arma::log_add_exp(sub_tree(5*d),sub_tree2(5*d));
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( 5*d+k+4),
                           sub_tree( 5*d+k+5),
                           sub_tree2(5*d+k+4),
                           sub_tree2(5*d+k+5));
      
      //next, check the termination condition       
      sub_tree(5*d+1) += 
        (sub_tree(5*d+k+4) - sub_tree(5*d) - log(1+sub_tree(5*d+k+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+2,5*d+2+k) / sub_tree(5*d+3+k),
                            Rcpp::Named("n") = sub_tree(5*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((4+K)*d+k+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-k-1))) + 
    arma::sum(arma::abs(sub_tree.subvec(2*d-k,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((4+K)*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((4+K)*d+k+6);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec((4+K)*d+1,(4+K)*d+3+k) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3+k);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree((4+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      //uniform sampling for the recycled values
      
      //cumulate the log multinomial weights
      sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
      
      //update alpha
      //alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      
      //check the condition of the virial:
      
      //cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+k+4),
                           sub_tree( (4+K)*d+k+5),
                           sub_tree2((4+K)*d+k+4),
                           sub_tree2((4+K)*d+k+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+k+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+k+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+2,(4+K)*d+2+k) / sub_tree((4+K)*d+3+k),
                            Rcpp::Named("n") = sub_tree((4+K)*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((4+K)*d+k+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-k-1)),M_inv_cont ) +
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((4+K)*d) = -H0;
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((4+K)*d+k+6);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec((4+K)*d+1,(4+K)*d+3+k) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3+k);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree((4+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      //uniform sampling for the recycled values
      
      //cumulate the log multinomial weights
      sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
      
      //update alpha
      //alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      

      //check the condition of the virial:
      
      //cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+k+4),
                           sub_tree( (4+K)*d+k+5),
                           sub_tree2((4+K)*d+k+4),
                           sub_tree2((4+K)*d+k+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+k+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+k+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+2,(4+K)*d+2+k) / sub_tree((4+K)*d+3+k),
                            Rcpp::Named("n") = sub_tree((4+K)*d+3+k),
                            Rcpp::Named("E") = H0);
}


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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((4+K)*d+k+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-k-1), M_inv_cont * sub_tree.subvec(d,2*d-k-1)) + 
    arma::dot(arma::abs(sub_tree.subvec(2*d-k,2*d-1)),M_inv_disc);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((4+K)*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((4+K)*d+k+6);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             k,
                             idx_disc,
                             log_tau,
                             M_inv_cont,
                             M_inv_disc,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec((4+K)*d+1,(4+K)*d+3+k) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3+k);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree((4+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      //uniform sampling for the recycled values
      
      //cumulate the log multinomial weights
      sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
      
      //update alpha
      //alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      

      //check the condition of the virial:
      
      //cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+k+4),
                           sub_tree( (4+K)*d+k+5),
                           sub_tree2((4+K)*d+k+4),
                           sub_tree2((4+K)*d+k+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+k+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+k+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+2,(4+K)*d+2+k) / sub_tree((4+K)*d+3+k),
                            Rcpp::Named("n") = sub_tree((4+K)*d+3+k),
                            Rcpp::Named("E") = H0);
}

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
                        const double& log_tau){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(5*d+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;

  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;

  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;

  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;

  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(5*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(5*d+6);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){

      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             log_tau);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{

      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             log_tau);

      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(5*d+1,5*d+3) += sub_tree2.subvec(5*d+1,5*d+3);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree(5*d+1)){
      if(arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //check the condition of the virial:
      
      //first, cumulate the log sum of the metropolis weights
      sub_tree(5*d) = arma::log_add_exp(sub_tree(5*d),sub_tree2(5*d));
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( 5*d+4),
                           sub_tree( 5*d+5),
                           sub_tree2(5*d+4),
                           sub_tree2(5*d+5));
      
      //next, check the termination condition       
      sub_tree(5*d+1) += 
        (sub_tree(5*d+4) - sub_tree(5*d) - log(1+sub_tree(5*d+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;

  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+2,5*d+2) / sub_tree(5*d+3),
                            Rcpp::Named("n") = sub_tree(5*d+3),
                            Rcpp::Named("E") = H0);
}

// diagonal matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau,
                        const arma::vec& M_inv){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(5*d+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;

  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;

  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;

  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;

  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-1)),M_inv);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(5*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(5*d+6);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){

      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{

      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv);

      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(5*d+1,5*d+3) += sub_tree2.subvec(5*d+1,5*d+3);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree(5*d+1)){
      if(arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //check the condition of the virial:
      
      //first, cumulate the log sum of the metropolis weights
      sub_tree(5*d) = arma::log_add_exp(sub_tree(5*d),sub_tree2(5*d));
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( 5*d+4),
                           sub_tree( 5*d+5),
                           sub_tree2(5*d+4),
                           sub_tree2(5*d+5));
      
      //next, check the termination condition       
      sub_tree(5*d+1) += 
        (sub_tree(5*d+4) - sub_tree(5*d) - log(1+sub_tree(5*d+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;

  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+2,5*d+2) / sub_tree(5*d+3),
                            Rcpp::Named("n") = sub_tree(5*d+3),
                            Rcpp::Named("E") = H0);
}

// dense matrix case
Rcpp::List nuts_singolo(const arma::vec& theta0,
                        const arma::vec& m0,
                        const Rcpp::Function& nlp,
                        const Rcpp::List& args,
                        const double& eps,
                        const unsigned int& max_treedepth,
                        const unsigned int& d,
                        const double& log_tau,
                        const arma::mat& M_inv){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(5*d+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;

  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;

  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;

  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;

  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-1),M_inv * sub_tree.subvec(d,2*d-1));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree(5*d) = -H0;

  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(5*d+6);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){

      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv);

      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{

      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv);

      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree.subvec(5*d+1,5*d+3) += sub_tree2.subvec(5*d+1,5*d+3);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree(5*d+1)){
      if(arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      }
      
      //check the condition of the virial:
      
      //first, cumulate the log sum of the metropolis weights
      sub_tree(5*d) = arma::log_add_exp(sub_tree(5*d),sub_tree2(5*d));
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( 5*d+4),
                           sub_tree( 5*d+5),
                           sub_tree2(5*d+4),
                           sub_tree2(5*d+5));
      
      //next, check the termination condition       
      sub_tree(5*d+1) += 
        (sub_tree(5*d+4) - sub_tree(5*d) - log(1+sub_tree(5*d+3)) ) < log_tau;
      
    }
    
    //increase the depth of the tree
    depth++;

  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+2,5*d+2) / sub_tree(5*d+3),
                            Rcpp::Named("n") = sub_tree(5*d+3),
                            Rcpp::Named("E") = H0);
}

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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((4+K)*d+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::sum(arma::square(sub_tree.subvec(d,2*d-1)));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((4+K)*d) = -H0;
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((4+K)*d+6);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((4+K)*d+1,(4+K)*d+3) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree((4+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      //uniform sampling for the recycled values
      
      //cumulate the log multinomial weights
      sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
      
      //update alpha
      //alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+4),
                           sub_tree( (4+K)*d+5),
                           sub_tree2((4+K)*d+4),
                           sub_tree2((4+K)*d+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+3)) ) < log_tau;
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+2,(4+K)*d+2) / sub_tree((4+K)*d+3),
                            Rcpp::Named("n") = sub_tree((4+K)*d+3),
                            Rcpp::Named("E") = H0);
}

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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((4+K)*d+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(arma::square(sub_tree.subvec(d,2*d-1)),M_inv);
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((4+K)*d) = -H0;
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((4+K)*d+6);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((4+K)*d+1,(4+K)*d+3) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree((4+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      //uniform sampling for the recycled values
      
      //cumulate the log multinomial weights
      sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
      
      //update alpha
      //alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      

      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+4),
                           sub_tree( (4+K)*d+5),
                           sub_tree2((4+K)*d+4),
                           sub_tree2((4+K)*d+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+3)) ) < log_tau;
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+2,(4+K)*d+2) / sub_tree((4+K)*d+3),
                            Rcpp::Named("n") = sub_tree((4+K)*d+3),
                            Rcpp::Named("E") = H0);
}

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
                        const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((4+K)*d+6);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }

  //calculate the current value of the Hamiltonian (energy level to be explored)
  double H0 = Rcpp::as<double>(nlp(sub_tree.subvec(4*d,5*d-1),args,true)) + 
    0.5 * arma::dot(sub_tree.subvec(d,2*d-1),M_inv * sub_tree.subvec(d,2*d-1));
  
  //initialize the logarithm of the sum weight for multinomial sampling from the trajectory
  sub_tree((4+K)*d) = -H0;
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((4+K)*d+6);
  
  //acceptance probability
  double alpha = 0.0;
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                             nlp,
                             args,
                             -eps,
                             depth,
                             H0,
                             d,
                             log_tau,
                             M_inv,
                             K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
      }
      
    }
    
    //cumulates metropolis acceptance rates and number of leaves
    sub_tree.subvec((4+K)*d+1,(4+K)*d+3) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3);
    
    //if we haven't encountered any disagreements, update the proposed value
    //with probability proportional to the weights of the trees
    if(!sub_tree((4+K)*d+1)){
      //biased sampling for the proposed value
      alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      // if(arma::randu() < alpha){
      //   sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(4*d,5*d-1);
      // }
      
      for(unsigned int i = 0; i < K; i++){
        if(arma::randu() < alpha){
          sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
        }
      }
      //uniform sampling for the recycled values
      
      //cumulate the log multinomial weights
      sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
      
      //update alpha
      //alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
      
      //sample from the trajectory
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+4),
                           sub_tree( (4+K)*d+5),
                           sub_tree2((4+K)*d+4),
                           sub_tree2((4+K)*d+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+3)) ) < log_tau;
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+2,(4+K)*d+2) / sub_tree((4+K)*d+3),
                            Rcpp::Named("n") = sub_tree((4+K)*d+3),
                            Rcpp::Named("E") = H0);
}

/* -------------------------------------------------------------------------- */

/* ------------------ VERSION WITH ONLY DISCRETE PARAMETERS ----------------- */

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
                         const double& tau){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+5);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree(5*d) = sub_tree(5*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree(5*d) + arma::sum(arma::abs(m0));
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+5);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              tau);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree(5*d + 2) = sub_tree2(5*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              tau);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree(5*d) = sub_tree2(5*d);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree(5*d+1) += sub_tree2(5*d+1);
    sub_tree.subvec(5*d+3,6*d+3) += sub_tree2.subvec(5*d+3,6*d+3);
    
    //if haven't encountered any divergences, update the trajectory proposed value
    if(!sub_tree(5*d+1)){
      //if(arma::randu() < 0.5){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      //}
      
      //then, cumulates the virial
      sub_tree(6*d + 4) += sub_tree2(6*d + 4);
      
      //and check the termination condition
      sub_tree(5*d+1) += 
        std::abs( sub_tree(6*d+4) / (1+sub_tree(6*d+3)) / sub_tree(6*d+3) ) < tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+3,6*d+2) / sub_tree(6*d+3),
                            Rcpp::Named("n") = sub_tree(6*d+3),
                            Rcpp::Named("E") = H0);
}

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
                         const arma::vec& M_inv){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>(6*d+5);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal value sampled from the tree
  sub_tree.subvec(4*d,5*d-1) = theta0;
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree(5*d) = sub_tree(5*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree(5*d) + arma::sum(M_inv % arma::abs(m0));
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>(6*d+5);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree(5*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              tau,
                              M_inv);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree(5*d + 2) = sub_tree2(5*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              tau,
                              M_inv);
      
      if(!sub_tree2(5*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree(5*d) = sub_tree2(5*d);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree(5*d+1) += sub_tree2(5*d+1);
    sub_tree.subvec(5*d+3,6*d+3) += sub_tree2.subvec(5*d+3,6*d+3);
    
    //if haven't encountered any divergences, update the trajectory proposed value
    if(!sub_tree(5*d+1)){
      //if(arma::randu() < 0.5){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      //}
      
      //then, cumulates the virial
      sub_tree(6*d + 4) += sub_tree2(6*d + 4);
      
      //and check the termination condition
      sub_tree(5*d+1) += 
        std::abs( sub_tree(6*d+4) / (1+sub_tree(6*d+3)) / sub_tree(6*d+3) ) < tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,5*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec(5*d+3,6*d+2) / sub_tree(6*d+3),
                            Rcpp::Named("n") = sub_tree(6*d+3),
                            Rcpp::Named("E") = H0);
}


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
                         const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+5);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree((4+K)*d) = sub_tree((4+K)*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree((4+K)*d) + arma::sum(arma::abs(m0));
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+5);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              tau,
                              K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree((4+K)*d + 2) = sub_tree2((4+K)*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              tau,
                              K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree((4+K)*d) = sub_tree2((4+K)*d);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree((4+K)*d+1) += sub_tree2((4+K)*d+1);
    sub_tree.subvec((4+K)*d+3,(5+K)*d+3) += sub_tree2.subvec((4+K)*d+3,(5+K)*d+3);
    
    //if haven't encountered any divergences, update the trajectory extreme value
    if(!sub_tree((4+K)*d+1)){
      
      //sample with probability one the proposed value
      //sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      sub_tree.subvec(4*d,(4+K)*d-1) = sub_tree2.subvec(4*d,(4+K)*d-1);
      
      //sample with probability 0.5 the recycled values
      // for(unsigned int i = 1; i < K; i++){
      //   if(arma::randu() < 0.5){
      //     sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
      //   }
      // }
      
      //then, cumulates the virial
      sub_tree((5+K)*d + 4) += sub_tree2((5+K)*d + 4);
      
      //and check the termination condition
      sub_tree((4+K)*d+1) += 
        std::abs( sub_tree((5+K)*d+4) / (1+sub_tree((5+K)*d+3)) / sub_tree((5+K)*d+3) ) < tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+3,(5+K)*d+2) / sub_tree((5+K)*d+3),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3),
                            Rcpp::Named("E") = H0);
}

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
                         const arma::vec& M_inv,
                         const unsigned int& K){
  
  //initialize the final tree
  arma::vec sub_tree = arma::zeros<arma::vec>((5+K)*d+5);
  
  //initial position of the particle at the left end of the tree
  sub_tree.subvec(0,d-1) = theta0;
  
  //initial momentum of the particle's momentum at the left end of the tree
  sub_tree.subvec(d,2*d-1) = m0;
  
  //initial position of the particle at the right end of the tree
  sub_tree.subvec(2*d,3*d-1) = theta0;
  
  //initial position of the particle momentum at the right end of the tree
  sub_tree.subvec(3*d,4*d-1) = m0;
  
  //proposal values sampled from the tree, K in total
  for(unsigned int i = 0; i < K; i++){
    sub_tree.subvec((4+i)*d,(5+i)*d - 1) = theta0;
  }
  
  //initialize the value of the potential energy both on the right and on the left
  sub_tree((4+K)*d) = sub_tree((4+K)*d+2) = Rcpp::as<double>(nlp(theta0,args,true));
  
  //we also compute the energy level
  double H0 = sub_tree((4+K)*d) + arma::sum(M_inv % arma::abs(m0));
  
  //depth of the current tree
  unsigned int depth = 0;
  
  //let's build the tree recursively
  
  //initialize the subtree which doubles the current one
  arma::vec sub_tree2 = arma::zeros<arma::vec>((5+K)*d+5);
  
  //until we met a stopping criterion, divergent transitions or
  //the maximum depth has been reached, continue to double the tree
  while(!sub_tree((4+K)*d + 1) && depth < max_treedepth){
    
    //should the doubling be done on the right or left?
    if(arma::randu() < 0.5){
      
      //double the tree on the right
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              eps,
                              depth,
                              d,
                              idx_disc,
                              tau,
                              M_inv,
                              K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the right)
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the right end
        sub_tree((4+K)*d + 2) = sub_tree2((4+K)*d + 2);
      }
      
    }else{
      
      //double the tree on the left
      sub_tree2 = build_tree(sub_tree,
                              nlp,
                              args,
                              -eps,
                              depth,
                              d,
                              idx_disc,
                              tau,
                              M_inv,
                              K);
      
      if(!sub_tree2((4+K)*d+1)){
        //update the extreme values of the trajectory (only on the left)
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(0,2*d-1);
        
        //update the U value of the left endpoint
        sub_tree((4+K)*d) = sub_tree2((4+K)*d);
      }
      
    }
    
    //accumulate the rest: metropolis acceptance rates and number of leaves
    sub_tree((4+K)*d+1) += sub_tree2((4+K)*d+1);
    sub_tree.subvec((4+K)*d+3,(5+K)*d+3) += sub_tree2.subvec((4+K)*d+3,(5+K)*d+3);
    
    //if haven't encountered any divergences, update the trajectory proposed value
    if(!sub_tree((4+K)*d+1)){
      //sample with probability one the proposed value
      //sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
      sub_tree.subvec(4*d,(4+K)*d-1) = sub_tree2.subvec(4*d,(4+K)*d-1);
      
      //sample with probability 0.5 the recycled values
      // for(unsigned int i = 1; i < K; i++){
      //   if(arma::randu() < 0.5){
      //     sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree2.subvec((4+i)*d,(5+i)*d-1);
      //   }
      // }
      //then, cumulates the virial
      sub_tree((5+K)*d + 4) += sub_tree2((5+K)*d + 4);
      
      //and check the termination condition
      sub_tree((4+K)*d+1) += 
        std::abs( sub_tree((5+K)*d+4) / (1+sub_tree((5+K)*d+3)) / sub_tree((5+K)*d+3) ) < tau;
      
    }
    
    //increase the depth of the tree
    depth++;
  }
  //return the proposed value, the average acceptance rate,
  //the length of the trajectory and the current energy level
  
  return Rcpp::List::create(Rcpp::Named("theta") = sub_tree.subvec(4*d,(4+K)*d-1),
                            Rcpp::Named("alpha") = sub_tree.subvec((4+K)*d+3,(5+K)*d+2) / sub_tree((5+K)*d+3),
                            Rcpp::Named("n") = sub_tree((5+K)*d+3),
                            Rcpp::Named("E") = H0);
}
