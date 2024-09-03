#include <iostream>
#include <RcppArmadillo.h>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "recursive_tree.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

// FUNCTIONS TO BUILD THE TRAJECTORY PROGRESSIVELY

/* -------------------------------- DNUTS ----------------------------------- */

// identity matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& k,
                     arma::uvec& idx_disc){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));
  
    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);
        
        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        sub_tree(6*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;

        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree(6*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-k-1))) - 
          arma::sum(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)));

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree(6*d))){
          sub_tree(6*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree(6*d) - H0) > 1000){

          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree(6*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

          //also set the value proposed by this leaf equal to the step taken
          sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);
          
          //cumulate the kinetic energy gradient
          sub_tree.subvec(5*d,6*d-k-1) = sub_tree.subvec(3*d,4*d-k-1);
          sub_tree.subvec(6*d-k,6*d-1) = arma::sign(sub_tree.subvec(4*d-k,4*d-1));
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree(6*d+2) = std::min(1.0,std::exp(H0+sub_tree(6*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree(6*d+3+k) = 1;

      }else{

        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree(6*d+1) = 1.0;      }
    }else{

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(6*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    
    //first build the tree adjacent to the point from which to start the doubling
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc);
    
    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(6*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,k);

      //cumulate the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(6*d+1) && arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);

      }
      
      //cumulate the remainder: log_sum_exp of the multinomial weights,
      //metropolis acceptance rates and number of leaves
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      sub_tree.subvec(6*d+1,6*d+3+k) += sub_tree2.subvec(6*d+1,6*d+3+k);

    }
    
  }
  
  //return the tree
  return sub_tree;
}


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
                     const arma::vec& M_inv_disc){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1+segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
  
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);
  
        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);
  
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
     
        //calculation of the Metropolis acceptance rate
        sub_tree(6*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;
       
        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
       
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
   
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;
       
        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree(6*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(arma::square(sub_tree.subvec(idx + d,idx + 2*d-k-1)),M_inv_cont ) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree(6*d))){
          sub_tree(6*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree(6*d) - H0) > 1000){
          
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree(6*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);
          
          //also set the value proposed by this leaf equal to the step taken
          sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);
          
          //cumulate the kinetic energy gradient
          sub_tree.subvec(5*d,6*d-k-1) = sub_tree.subvec(3*d,4*d-k-1);
          sub_tree.subvec(6*d-k,6*d-1) = arma::sign(sub_tree.subvec(4*d-k,4*d-1));
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree(6*d+2) = std::min(1.0,std::exp(H0+sub_tree(6*d)));
        
        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree(6*d+3+k) = 1;
        
      }else{
        
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree(6*d+1) = 1.0;      
      }
    }else{
      
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(6*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
    
    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(6*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,k,M_inv_cont,M_inv_disc);
      
      //cumulate the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(6*d+1) && arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
       
      }
      
      //cumulate the remainder: log_sum_exp of the multinomial weights,
      //metropolis acceptance rates and number of leaves
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      sub_tree.subvec(6*d+1,6*d+3+k) += sub_tree2.subvec(6*d+1,6*d+3+k);
      
    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const arma::vec& M_inv_disc){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1+segno(eps)) * d;
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));
    
    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));
    
    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);
        
        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        sub_tree(6*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));
        
        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;
          
        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
      
      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;
        
        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree(6*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(sub_tree.subvec(idx + d,idx + 2*d-k-1), M_inv_cont * sub_tree.subvec(idx + d,idx + 2*d-k-1)) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree(6*d))){
          sub_tree(6*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree(6*d) - H0) > 1000){
          
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree(6*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);
          
          //also set the value proposed by this leaf equal to the step taken
          sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);
          
          //cumulate the kinetic energy gradient
          sub_tree.subvec(5*d,6*d-k-1) = sub_tree.subvec(3*d,4*d-k-1);
          sub_tree.subvec(6*d-k,6*d-1) = arma::sign(sub_tree.subvec(4*d-k,4*d-1));
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree(6*d+2) = std::min(1.0,std::exp(H0+sub_tree(6*d)));
        
        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree(6*d+3+k) = 1;
        
        
      }else{
        
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree(6*d+1) = 1.0;      }
    }else{
      
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(6*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
    
    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(6*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc);
      
      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,k,M_inv_cont,M_inv_disc);
      
      //cumulate the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(6*d+1) && arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);
        
      }
      
      //cumulate the remainder: log_sum_exp of the multinomial weights,
      //metropolis acceptance rates and number of leaves
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      sub_tree.subvec(6*d+1,6*d+3+k) += sub_tree2.subvec(6*d+1,6*d+3+k);
      
    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));
    
    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));
    
    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);
        
        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        sub_tree((5+K)*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));
        
        //refraction or reflection?
        if( std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
          U += delta_U;
          
        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
      
      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;
        
        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree((5+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-k-1))) - 
          arma::sum(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)));
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree((5+K)*d))){
          sub_tree((5+K)*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree((5+K)*d) - H0) > 1000){
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree((5+K)*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);
          
          //also set the value proposed by this leaf equal to the step taken
          //for each recycled sample
          for(unsigned int i = 0; i < K; i++){
            sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
          }
          
          //cumulate the kinetic energy gradient
          sub_tree.subvec((4+K)*d,(5+K)*d-k-1) = sub_tree.subvec(3*d,4*d-k-1);
          sub_tree.subvec((5+K)*d-k,(5+K)*d-1) = arma::sign(sub_tree.subvec(4*d-k,4*d-1));
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree((5+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((5+K)*d)));
        
        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree((5+K)*d+3+k) = 1;
        
      }else{
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree((5+K)*d+1) = 1.0;      }
    }else{
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,K);
    
    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,K);
      
      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,k,K);
      
      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((5+K)*d+1)){
        double alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((5+K)*d+1,(5+K)*d+3+k) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3+k);

    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1+segno(eps)) * d;
   
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));
    
    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));
    
    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);
        
        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        sub_tree((5+K)*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));
        
        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;
          
        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
      
      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;
        
        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree((5+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(arma::square(sub_tree.subvec(idx + d,idx + 2*d-k-1)),M_inv_cont ) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree((5+K)*d))){
          sub_tree((5+K)*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree((5+K)*d) - H0) > 1000){
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree((5+K)*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);
          
          //also set the value proposed by this leaf equal to the step taken
          //for each recycled sample
          for(unsigned int i = 0; i < K; i++){
            sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
          }
          
          //cumulate the kinetic energy gradient
          sub_tree.subvec((4+K)*d,(5+K)*d-k-1) = sub_tree.subvec(3*d,4*d-k-1);
          
          sub_tree.subvec((5+K)*d-k,(5+K)*d-1) = arma::sign(sub_tree.subvec(4*d-k,4*d-1));
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree((5+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((5+K)*d)));
        
        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree((5+K)*d+3+k) = 1;
        
      }else{
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree((5+K)*d+1) = 1.0;      }
    }else{
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc,K);
    
    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc,K);
      
      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,k,M_inv_cont,M_inv_disc,K);
      
      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((5+K)*d+1)){
        double alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((5+K)*d+1,(5+K)*d+3+k) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3+k);
      
    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1+segno(eps)) * d;
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));
    
    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
    
    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));
    
    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);
        
        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);
        
        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
        
        //calculation of the Metropolis acceptance rate
        sub_tree((5+K)*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));
        
        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;
          
        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
      
      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;
        
        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree((5+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(sub_tree.subvec(idx + d,idx + 2*d-k-1), M_inv_cont * sub_tree.subvec(idx + d,idx + 2*d-k-1)) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree((5+K)*d))){
          sub_tree((5+K)*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree((5+K)*d) - H0) > 1000){
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree((5+K)*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);
          
          //also set the value proposed by this leaf equal to the step taken
          //for each recycled sample
          for(unsigned int i = 0; i < K; i++){
            sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
          }
          
          //cumulate the kinetic energy gradient
          sub_tree.subvec((4+K)*d,(5+K)*d-k-1) = sub_tree.subvec(3*d,4*d-k-1);
          sub_tree.subvec((5+K)*d-k,(5+K)*d-1) = arma::sign(sub_tree.subvec(4*d-k,4*d-1));
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree((5+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((5+K)*d)));
        
        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree((5+K)*d+3+k) = 1;
        
      }else{
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree((5+K)*d+1) = 1.0;      }
    }else{
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc,K);
    
    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,M_inv_cont,M_inv_disc,K);
      
      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,k,M_inv_cont,M_inv_disc,K);
      
      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((5+K)*d+1)){
        double alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((5+K)*d+1,(5+K)*d+3+k) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3+k);
      
    }
    
  }
  //return the tree
  return sub_tree;
}

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
                     const unsigned int& d){

  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));
    
    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree(6*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1))); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree(6*d))){
      sub_tree(6*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree(6*d) - H0) > 1000){

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(6*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

      //cumulate the kinetic energy gradient
      sub_tree.subvec(5*d,6*d-1) = sub_tree.subvec(3*d,4*d-1);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree(6*d+2) = std::min(1.0,std::exp(H0+sub_tree(6*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(6*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(6*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d);

      //cumulate the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(6*d+1) && arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);

      }
      
      //cumulate the remainder: log_sum_exp of the multinomial weights,
      //metropolis acceptance rates and number of leaves
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      sub_tree.subvec(6*d+1,6*d+3) += sub_tree2.subvec(6*d+1,6*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::vec& M_inv){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree(6*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1)),M_inv); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree(6*d))){
      sub_tree(6*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree(6*d) - H0) > 1000){

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(6*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

      //cumulate the kinetic energy gradient
      sub_tree.subvec(5*d,6*d-1) = sub_tree.subvec(3*d,4*d-1);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree(6*d+2) = std::min(1.0,std::exp(H0+sub_tree(6*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(6*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(6*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,M_inv);

      //cumulate the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(6*d+1) && arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);

      }
      
      //cumulate the remainder: log_sum_exp of the multinomial weights,
      //metropolis acceptance rates and number of leaves
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      sub_tree.subvec(6*d+1,6*d+3) += sub_tree2.subvec(6*d+1,6*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// dense matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::mat& M_inv){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree(6*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(sub_tree.subvec(idx + d ,idx + 2*d-1),M_inv * sub_tree.subvec(idx + d ,idx + 2*d-1)); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree(6*d))){
      sub_tree(6*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree(6*d) - H0) > 1000){

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(6*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

      //cumulate the kinetic energy gradient
      sub_tree.subvec(5*d,6*d-1) = sub_tree.subvec(3*d,4*d-1);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree(6*d+2) = std::min(1.0,std::exp(H0+sub_tree(6*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(6*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(6*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(6*d+1) += check_u_turn(sub_tree,d,M_inv);

      //cumulate the kinetic energy gradients
      sub_tree.subvec(5*d,6*d-1) += sub_tree2.subvec(5*d,6*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(6*d+1) && arma::randu() < std::exp(sub_tree2(6*d) - sub_tree(6*d))){
        sub_tree.subvec(4*d,5*d-1) = sub_tree2.subvec(4*d,5*d-1);

      }
      
      //cumulate the remainder: log_sum_exp of the multinomial weights,
      //metropolis acceptance rates and number of leaves
      sub_tree(6*d) = arma::log_add_exp(sub_tree(6*d),sub_tree2(6*d));
      sub_tree.subvec(6*d+1,6*d+3) += sub_tree2.subvec(6*d+1,6*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

/* --------------------------   RECYCLED VERSION   -------------------------- */

// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree((5+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1))); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree((5+K)*d))){
      sub_tree((5+K)*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree((5+K)*d) - H0) > 1000){
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //cumulate the kinetic energy gradient
      sub_tree.subvec((4+K)*d,(5+K)*d-1) = sub_tree.subvec(3*d,4*d-1);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree((5+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((5+K)*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((5+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,K);

      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((5+K)*d+1)){
        double alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
        
        //recalculate the probability before uniform sampling
        alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((5+K)*d+1,(5+K)*d+3) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree((5+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1)),M_inv); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree((5+K)*d))){
      sub_tree((5+K)*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree((5+K)*d) - H0) > 1000){
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //cumulate the kinetic energy gradient
      sub_tree.subvec((4+K)*d,(5+K)*d-1) = sub_tree.subvec(3*d,4*d-1);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree((5+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((5+K)*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((5+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,M_inv,K);

      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((5+K)*d+1)){
        double alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
        
        //recalculate the probability before uniform sampling
        alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((5+K)*d+1,(5+K)*d+3) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// dense matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const arma::mat& M_inv,
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree((5+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(sub_tree.subvec(idx + d ,idx + 2*d-1),M_inv * sub_tree.subvec(idx + d ,idx + 2*d-1) ); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree((5+K)*d))){
      sub_tree((5+K)*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree((5+K)*d) - H0) > 1000){
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //cumulate the kinetic energy gradient
      sub_tree.subvec((4+K)*d,(5+K)*d-1) = sub_tree.subvec(3*d,4*d-1);
      
    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree((5+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((5+K)*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((5+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,M_inv,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec(sub_tree,d,M_inv,K);

      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((5+K)*d+1)){
        double alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((5+K)*d) = arma::log_add_exp(sub_tree((5+K)*d),sub_tree2((5+K)*d));
        
        //recalculate the probability before uniform sampling
        alpha = std::exp(sub_tree2((5+K)*d) - sub_tree((5+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((5+K)*d+1,(5+K)*d+3) += sub_tree2.subvec((5+K)*d+1,(5+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

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
                     arma::uvec& idx_disc){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //compute the value of the new potential energy
    double U = sub_tree(5*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);

    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree(5*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }

    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree(6*d + 4) = segno(eps);

      //cumulate the kinetic energy gradient
      sub_tree.subvec(4*d,5*d-1) = arma::sign(sub_tree.subvec(3*d,4*d-1));

      //update the extreme value of U on the trajectory
      sub_tree(5*d + 1 + segno(eps)) = U;

    }

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(6*d+3) = 1;
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree(5*d + 2) = sub_tree2(5*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the left endpoint
        sub_tree(5*d) = sub_tree2(5*d);
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(5*d+1) += check_u_turn2(sub_tree,d);

      //cumulate the kinetic energy gradients
      sub_tree.subvec(4*d,5*d-1) += sub_tree2.subvec(4*d,5*d-1);

      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      sub_tree(6*d+4) = sub_tree2(6*d+4);

      //cumulate the remainder: metropolis acceptance rates and number of leaves
      sub_tree(5*d+1) += sub_tree2(5*d+1);
      sub_tree.subvec(5*d+3,6*d+3) += sub_tree2.subvec(5*d+3,6*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const arma::vec& M_inv){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //compute the value of the new potential energy
    double U = sub_tree(5*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);

    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv(j-idx);

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree(5*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( M_inv(j-idx) * std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv(j-idx);
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }
    
    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree(6*d + 4) = segno(eps);

      //cumulate the kinetic energy gradient
      sub_tree.subvec(4*d,5*d-1) = arma::sign(sub_tree.subvec(3*d,4*d-1));
      
      //update the extreme value of U on the trajectory
      sub_tree(5*d + 1 + segno(eps)) = U;
      
    }
    
    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(6*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,M_inv);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,M_inv);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree(5*d + 2) = sub_tree2(5*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the left endpoint
        sub_tree(5*d) = sub_tree2(5*d);
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree(5*d+1) += check_u_turn2(sub_tree,d,M_inv);

      //cumulate the kinetic energy gradients
      sub_tree.subvec(4*d,5*d-1) += sub_tree2.subvec(4*d,5*d-1);

      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      
      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      
      sub_tree(6*d+4) = sub_tree2(6*d+4);

      //cumulate the remainder: metropolis acceptance rates and number of leaves
      sub_tree(5*d+1) += sub_tree2(5*d+1);
      sub_tree.subvec(5*d+3,6*d+3) += sub_tree2.subvec(5*d+3,6*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

/* -------------------------- RECYCLED VERSION ------------------------------ */


// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //compute the value of the new potential energy
    double U = sub_tree((5+K)*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    
    
    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree((5+K)*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }

    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree((6+K)*d + 4) = segno(eps);

      //cumulate the kinetic energy gradient
      sub_tree.subvec((4+K)*d,(5+K)*d-1) = arma::sign(sub_tree.subvec(3*d,4*d-1));

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //update the extreme value of U on the trajectory
      sub_tree((5+K)*d + 1 + segno(eps)) = U;
      
    }

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((6+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree((5+K)*d + 2) = sub_tree2((5+K)*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the left endpoint
        sub_tree((5+K)*d) = sub_tree2((5+K)*d);
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec2(sub_tree,d,K+1);

      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);

      //update the value of theta_prop based on the ratio
      //this time we do uniform sampling without bias
      //otherwise we would always have only the extreme value recycled
      if(!sub_tree((5+K)*d+1)){
        for(unsigned int i = 0; i<K;i++){
          if(arma::randu() < 0.5){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      sub_tree((6+K)*d+4) = sub_tree2((6+K)*d+4);

      //cumulate the remainder: metropolis acceptance rates and number of leaves
      sub_tree((5+K)*d+1) += sub_tree2((5+K)*d+1);
      sub_tree.subvec((5+K)*d+3,(6+K)*d+3) += sub_tree2.subvec((5+K)*d+3,(6+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const arma::vec& M_inv,
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //compute the value of the new potential energy
    double U = sub_tree((5+K)*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);

    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv(j-idx);

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree((5+K)*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( M_inv(j-idx) * std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv(j-idx);
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;
        
      }
      
    }
    
    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((5+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree((6+K)*d + 4) = segno(eps);

      //cumulate the kinetic energy gradient
      sub_tree.subvec((4+K)*d,(5+K)*d-1) = arma::sign(sub_tree.subvec(3*d,4*d-1));

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //update the extreme value of U on the trajectory
      sub_tree((5+K)*d + 1 + segno(eps)) = U;
      
    }

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((6+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,M_inv,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((5+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,M_inv,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree((5+K)*d + 2) = sub_tree2((5+K)*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the left endpoint
        sub_tree((5+K)*d) = sub_tree2((5+K)*d);
      }
      
      //check the U-TURN or divergence condition with the second tree
      sub_tree((5+K)*d+1) += check_u_turn_rec2(sub_tree,d,M_inv,K+1);

      //cumulate the kinetic energy gradients
      sub_tree.subvec((4+K)*d,(5+K)*d-1) += sub_tree2.subvec((4+K)*d,(5+K)*d-1);

      //update the value of theta_prop based on the ratio
      //this time we do uniform sampling without bias
      //otherwise we would always have only the extreme value recycled
      if(!sub_tree((5+K)*d+1)){
        for(unsigned int i = 0; i<K;i++){
          if(arma::randu() < 0.5){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      
      sub_tree((6+K)*d+4) = sub_tree2((6+K)*d+4);

      //cumulate the remainder: metropolis acceptance rates and number of leaves
      sub_tree((5+K)*d+1) += sub_tree2((5+K)*d+1);
      sub_tree.subvec((5+K)*d+3,(6+K)*d+3) += sub_tree2.subvec((5+K)*d+3,(6+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

/* -------------------------------- XDHMC ----------------------------------- */

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
                     const double& log_tau){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //initialize the value of the virial
    sub_tree(5*d + k + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);

      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);

        //modify the discrete parameter
        theta_old = sub_tree(j);

        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));

        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

        //calculation of the Metropolis acceptance rate
        sub_tree(5*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;

        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree(5*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-k-1))) - 
          arma::sum(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)));

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree(5*d))){
          sub_tree(5*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree(5*d) - H0) > 1000){

          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree(5*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

          //also set the value proposed by this leaf equal to the step taken
          sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

          //update the value of the virial exchange rate
          sub_tree(5*d+k+4) = (sub_tree(5*d+k+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
          
          //calculate the sign of the virial exchange rate
          sub_tree(5*d+k+5) = segno(sub_tree(5*d+k+4));
          
          //calculate the logarithm of the virial exchange rate
          //multiplied by the multinomial weight
          sub_tree(5*d+k+4) = std::log(std::abs(sub_tree(5*d+k+4))) + sub_tree(5*d);
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree(5*d+2) = std::min(1.0,std::exp(H0+sub_tree(5*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree(5*d+3+k) = 1;

      }else{

        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree(5*d+1) = 1.0;      }
    }else{

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc, log_tau);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(5*d+1) && arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
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

      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree.subvec(5*d+1,5*d+3+k) += sub_tree2.subvec(5*d+1,5*d+3+k);

    }
    
  }
  //return the tree
  return sub_tree;
}


/*----------------------------------------------------------------------------*/

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
                     const double& log_tau,
                     const arma::vec& M_inv_cont,
                     const arma::vec& M_inv_disc){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //initialize the value of the virial
    sub_tree(5*d + k + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);

      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);

        //modify the discrete parameter
        theta_old = sub_tree(j);

        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);

        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

        //calculation of the Metropolis acceptance rate
        sub_tree(5*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree(5*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(arma::square(sub_tree.subvec(idx + d,idx + 2*d-k-1)),M_inv_cont ) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree(5*d))){
          sub_tree(5*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree(5*d) - H0) > 1000){

          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree(5*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

          //also set the value proposed by this leaf equal to the step taken
          sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

          //update the value of the virial exchange rate
          sub_tree(5*d+k+4) = (sub_tree(5*d+k+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
          
          //calculate the sign of the virial exchange rate
          sub_tree(5*d+k+5) = segno(sub_tree(5*d+k+4));
          
          //calculate the logarithm of the virial exchange rate
          //multiplied by the multinomial weight
          sub_tree(5*d+k+4) = std::log(std::abs(sub_tree(5*d+k+4))) + sub_tree(5*d);
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree(5*d+2) = std::min(1.0,std::exp(H0+sub_tree(5*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree(5*d+3+k) = 1;

      }else{

        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree(5*d+1) = 1.0;      }
    }else{

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,M_inv_cont,M_inv_disc);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc, log_tau,M_inv_cont,M_inv_disc);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(5*d+1) && arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
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

      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree.subvec(5*d+1,5*d+3+k) += sub_tree2.subvec(5*d+1,5*d+3+k);

    }
    
  }
  //return the tree
  return sub_tree;
}

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
                     const arma::vec& M_inv_disc){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //initialize the value of the virial
    sub_tree(5*d + k + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);

      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);

        //modify the discrete parameter
        theta_old = sub_tree(j);

        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);

        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

        //calculation of the Metropolis acceptance rate
        sub_tree(5*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;

        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));
      
      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree(5*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(sub_tree.subvec(idx + d,idx + 2*d-k-1), M_inv_cont * sub_tree.subvec(idx + d,idx + 2*d-k-1)) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);
        
        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree(5*d))){
          sub_tree(5*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree(5*d) - H0) > 1000){

          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree(5*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

          //also set the value proposed by this leaf equal to the step taken
          sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

          //update the value of the virial exchange rate
          sub_tree(5*d+k+4) = (sub_tree(5*d+k+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
          
          //calculate the sign of the virial exchange rate
          sub_tree(5*d+k+5) = segno(sub_tree(5*d+k+4));
          
          //calculate the logarithm of the virial exchange rate
          //multiplied by the multinomial weight
          sub_tree(5*d+k+4) = std::log(std::abs(sub_tree(5*d+k+4))) + sub_tree(5*d);
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree(5*d+2) = std::min(1.0,std::exp(H0+sub_tree(5*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree(5*d+3+k) = 1;

      }else{
        
        
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree(5*d+1) = 1.0;      }
    }else{
      
      
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,M_inv_cont,M_inv_disc);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc, log_tau,M_inv_cont,M_inv_disc);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(5*d+1) && arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
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
      
      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree.subvec(5*d+1,5*d+3+k) += sub_tree2.subvec(5*d+1,5*d+3+k);

    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);

        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));

        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;
 
        //calculation of the Metropolis acceptance rate
        sub_tree((4+K)*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;
          
        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree((4+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-k-1))) - 
          arma::sum(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)));

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree((4+K)*d))){
          sub_tree((4+K)*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree((4+K)*d) - H0) > 1000){
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree((4+K)*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);
 
          //also set the value proposed by this leaf equal to the step taken
          //for each recycled sample
          for(unsigned int i = 0; i < K; i++){
            sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
          }
          
          //update the value of the virial exchange rate
          sub_tree((4+K)*d+k+4) = (sub_tree((4+K)*d+k+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
          
          //calculate the sign of the virial exchange rate
          sub_tree((4+K)*d+k+5) = segno(sub_tree((4+K)*d+k+4));
          
          //calculate the logarithm of the virial exchange rate
          //multiplied by the multinomial weight
          sub_tree((4+K)*d+k+4) = std::log(std::abs(sub_tree((4+K)*d+k+4))) + sub_tree((4+K)*d);
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree((4+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((4+K)*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree((4+K)*d+3+k) = 1;

      }else{
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree((4+K)*d+1) = 1.0;      }
    }else{
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((4+K)*d+1)){
        double alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+k+4),
                           sub_tree( (4+K)*d+k+5),
                           sub_tree2((4+K)*d+k+4),
                           sub_tree2((4+K)*d+k+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+k+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+k+3)) ) < log_tau;
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((4+K)*d+1,(4+K)*d+3+k) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3+k);

    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);

        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);

        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

        //calculation of the Metropolis acceptance rate
        sub_tree((4+K)*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;

        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree((4+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(arma::square(sub_tree.subvec(idx + d,idx + 2*d-k-1)),M_inv_cont ) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree((4+K)*d))){
          sub_tree((4+K)*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree((4+K)*d) - H0) > 1000){
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree((4+K)*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

          //also set the value proposed by this leaf equal to the step taken
          //for each recycled sample
          for(unsigned int i = 0; i < K; i++){
            sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
          }
          
          //update the value of the virial exchange rate
          sub_tree((4+K)*d+k+4) = (sub_tree((4+K)*d+k+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
          
          //calculate the sign of the virial exchange rate
          sub_tree((4+K)*d+k+5) = segno(sub_tree((4+K)*d+k+4));
          
          //calculate the logarithm of the virial exchange rate
          //multiplied by the multinomial weight
          sub_tree((4+K)*d+k+4) = std::log(std::abs(sub_tree((4+K)*d+k+4))) + sub_tree((4+K)*d);
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree((4+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((4+K)*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree((4+K)*d+3+k) = 1;

      }else{
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree((4+K)*d+1) = 1.0;      }
    }else{
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont % sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,M_inv_cont,M_inv_disc,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,M_inv_cont,M_inv_disc,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((4+K)*d+1)){
        double alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+k+4),
                           sub_tree( (4+K)*d+k+5),
                           sub_tree2((4+K)*d+k+4),
                           sub_tree2((4+K)*d+k+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+k+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+k+3)) ) < log_tau;
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((4+K)*d+1,(4+K)*d+3+k) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3+k);

    }
    
  }
  //return the tree
  return sub_tree;
}



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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //continuous parameter update by half step size
    sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);

    //compute the value of the new potential energy
    double U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d - 1),args,true));

    // if the potential energy is finite then we continue
    if(arma::is_finite(U)){
      
      //initialization of the old value and the potential difference
      double theta_old;
      double delta_U;
      
      //permute the order of the discrete parameters
      idx_disc = arma::shuffle(idx_disc);
      unsigned int j;
      //loop for every discontinuous component
      for(unsigned int i = 0; i < k; i++){
        
        //set the current index
        j = idx + idx_disc(i);

        //modify the discrete parameter
        theta_old = sub_tree(j);
        
        sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv_disc(j-idx-d+k);

        //calculation of the difference in potential energy
        delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

        //calculation of the Metropolis acceptance rate
        sub_tree((4+K)*d+3+j-idx-d+k) = std::min(1.0,std::exp(-delta_U));

        //refraction or reflection?
        if( M_inv_disc(j-idx-d+k) * std::abs(sub_tree(d+j)) > delta_U ){
          
          //refraction
          sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv_disc(j-idx-d+k);
          U += delta_U;

        }else{
          
          //reflection
          sub_tree(j) = theta_old;
          sub_tree(d+j) *= -1.0;

        }
        
      }
      
      // continue updating continuous parameters
      
      //continuous parameter update by half step size
      sub_tree.subvec(idx,idx+d-k-1) += 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);

      //compute the gradient
      arma::vec grad = Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

      //let's make sure it's finished
      if(arma::is_finite(grad)){
        //continuous momentum update by half step size
        sub_tree.subvec(idx + d,idx +2*d-k-1) -= 0.5 * eps * grad;

        //we calculate the contribution to the sum of the metropolis log weights
        sub_tree((4+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
          0.5 * arma::dot(sub_tree.subvec(idx + d,idx + 2*d-k-1), M_inv_cont * sub_tree.subvec(idx + d,idx + 2*d-k-1)) - 
          arma::dot(arma::abs(sub_tree.subvec(idx + 2*d-k,idx + 2*d-1)),M_inv_disc);

        //let's make sure it's not NaN, in which case let's set it equal to -Inf
        if(!arma::is_finite(sub_tree((4+K)*d))){
          sub_tree((4+K)*d) = -arma::datum::inf;
        }
        
        //let's check if there is a divergent transition
        if( (-sub_tree((4+K)*d) - H0) > 1000){
          //add the divergent transition to the global matrix
          sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
          add_div_trans(sub_tree.subvec(idx,idx+d-1));
          
          //communicate that a stopping criterion has been met
          sub_tree((4+K)*d+1) = 1.0;
        }else{
          //if there is no divergent transition we also copy the other values
          
          //since we are at the deepest level of the tree
          //set the left extremes equal to the right ones or vice versa
          sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

          //also set the value proposed by this leaf equal to the step taken
          //for each recycled sample
          for(unsigned int i = 0; i < K; i++){
            sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
          }
          
          //update the value of the virial exchange rate
          sub_tree((4+K)*d+k+4) = (sub_tree((4+K)*d+k+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
          
          //calculate the sign of the virial exchange rate
          sub_tree((4+K)*d+k+5) = segno(sub_tree((4+K)*d+k+4));
          
          //calculate the logarithm of the virial exchange rate
          //multiplied by the multinomial weight
          sub_tree((4+K)*d+k+4) = std::log(std::abs(sub_tree((4+K)*d+k+4))) + sub_tree((4+K)*d);
          
        }
        
        //compute the metropolis acceptance rate, which has the mere purpose of tunin
        sub_tree((4+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((4+K)*d)));

        //initialize the count of the number of integrations made (leaves of the tree)
        sub_tree((4+K)*d+3+k) = 1;
        
      }else{
        //add the divergent transition to the global matrix
        sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
        add_div_trans(sub_tree.subvec(idx,idx+d-1));
        
        //communicate that a stopping criterion has been met
        sub_tree((4+K)*d+1) = 1.0;      }
    }else{
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-k-1) -= 0.5 * eps * M_inv_cont * sub_tree.subvec(idx + d,idx +2*d-k-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,M_inv_cont,M_inv_disc,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,k,idx_disc,log_tau,M_inv_cont,M_inv_disc,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((4+K)*d+1)){
        double alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+k+4),
                           sub_tree( (4+K)*d+k+5),
                           sub_tree2((4+K)*d+k+4),
                           sub_tree2((4+K)*d+k+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+k+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+k+3)) ) < log_tau;
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((4+K)*d+1,(4+K)*d+3+k) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3+k);
      
    }
    
  }
  //return the tree
  return sub_tree;
}

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
                     const double& log_tau){

  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree(5*d + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));
    
    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * sub_tree.subvec(idx + d,idx +2*d-1);
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree(5*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1))); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree(5*d))){
      sub_tree(5*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree(5*d) - H0) > 1000){

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

      //update the value of the virial exchange rate
      sub_tree(5*d+4) = (sub_tree(5*d+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
      
      //calculate the sign of the virial exchange rate
      sub_tree(5*d+5) = segno(sub_tree(5*d+4));
      
      //calculate the logarithm of the virial exchange rate
      //multiplied by the multinomial weight
      sub_tree(5*d+4) = std::log(std::abs(sub_tree(5*d+4))) + sub_tree(5*d);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree(5*d+2) = std::min(1.0,std::exp(H0+sub_tree(5*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(5*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(5*d+1) && arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
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

      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree.subvec(5*d+1,5*d+3) += sub_tree2.subvec(5*d+1,5*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const arma::vec& M_inv){
  
  
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree(5*d + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree(5*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1)),M_inv); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree(5*d))){
      sub_tree(5*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree(5*d) - H0) > 1000){

      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

      //update the value of the virial exchange rate
      sub_tree(5*d+4) = (sub_tree(5*d+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
      
      //calculate the sign of the virial exchange rate
      sub_tree(5*d+5) = segno(sub_tree(5*d+4));
      
      //calculate the logarithm of the virial exchange rate
      //multiplied by the multinomial weight
      sub_tree(5*d+4) = std::log(std::abs(sub_tree(5*d+4))) + sub_tree(5*d);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree(5*d+2) = std::min(1.0,std::exp(H0+sub_tree(5*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(5*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(5*d+1) && arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
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

      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree.subvec(5*d+1,5*d+3) += sub_tree2.subvec(5*d+1,5*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

// dense matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const double& H0,
                     const unsigned int& d,
                     const double& log_tau,
                     const arma::mat& M_inv){

  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree(5*d + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree(5*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(sub_tree.subvec(idx + d ,idx + 2*d-1),M_inv * sub_tree.subvec(idx + d ,idx + 2*d-1)); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree(5*d))){
      sub_tree(5*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree(5*d) - H0) > 1000){
      
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(5*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      sub_tree.subvec(4*d,5*d-1) = sub_tree.subvec(0,d-1);

      //update the value of the virial exchange rate
      sub_tree(5*d+4) = (sub_tree(5*d+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
      
      //calculate the sign of the virial exchange rate
      sub_tree(5*d+5) = segno(sub_tree(5*d+4));
      
      //calculate the logarithm of the virial exchange rate
      //multiplied by the multinomial weight
      sub_tree(5*d+4) = std::log(std::abs(sub_tree(5*d+4))) + sub_tree(5*d);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree(5*d+2) = std::min(1.0,std::exp(H0+sub_tree(5*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(5*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(5*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree(5*d+1) && arma::randu() < std::exp(sub_tree2(5*d) - sub_tree(5*d))){
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
      
      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree.subvec(5*d+1,5*d+3) += sub_tree2.subvec(5*d+1,5*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree((4+K)*d + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * sub_tree.subvec(idx + d,idx +2*d-1);
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree((4+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::sum(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1))); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree((4+K)*d))){
      sub_tree((4+K)*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree((4+K)*d) - H0) > 1000){
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //update the value of the virial exchange rate
      sub_tree((4+K)*d+4) = (sub_tree((4+K)*d+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
      
      //calculate the sign of the virial exchange rate
      sub_tree((4+K)*d+5) = segno(sub_tree((4+K)*d+4));
      
      //calculate the logarithm of the virial exchange rate
      //multiplied by the multinomial weight
      sub_tree((4+K)*d+4) = std::log(std::abs(sub_tree((4+K)*d+4))) + sub_tree((4+K)*d);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree((4+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((4+K)*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((4+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((4+K)*d+1)){
        double alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+4),
                           sub_tree( (4+K)*d+5),
                           sub_tree2((4+K)*d+4),
                           sub_tree2((4+K)*d+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+3)) ) < log_tau;
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((4+K)*d+1,(4+K)*d+3) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}


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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree((4+K)*d + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree((4+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(arma::square(sub_tree.subvec(idx + d ,idx + 2*d-1)),M_inv); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree((4+K)*d))){
      sub_tree((4+K)*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree((4+K)*d) - H0) > 1000){
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv % sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }
      
      //update the value of the virial exchange rate
      sub_tree((4+K)*d+4) = (sub_tree((4+K)*d+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
      
      //calculate the sign of the virial exchange rate
      sub_tree((4+K)*d+5) = segno(sub_tree((4+K)*d+4));
      
      //calculate the logarithm of the virial exchange rate
      //multiplied by the multinomial weight
      sub_tree((4+K)*d+4) = std::log(std::abs(sub_tree((4+K)*d+4))) + sub_tree((4+K)*d);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree((4+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((4+K)*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((4+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv,K);
      
      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((4+K)*d+1)){
        double alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+4),
                           sub_tree( (4+K)*d+5),
                           sub_tree2((4+K)*d+4),
                           sub_tree2((4+K)*d+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+3)) ) < log_tau;
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((4+K)*d+1,(4+K)*d+3) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

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
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){
    
    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree((4+K)*d + 4) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d -1),args,false));

    //update the continuous parameter by one step size
    sub_tree.subvec(idx,idx+d-1) +=  eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);
    
    //continuous momentum update by half step size
    sub_tree.subvec(idx + d,idx +2*d-1) -= 0.5 * eps * Rcpp::as<arma::vec>(nlp(sub_tree.subvec(idx,idx + d-1),args,false));

    //we calculate the contribution to the sum of the metropolis log weights
    sub_tree((4+K)*d) = -Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx + d-1),args,true)) - 
      0.5*arma::dot(sub_tree.subvec(idx + d ,idx + 2*d-1),M_inv * sub_tree.subvec(idx + d ,idx + 2*d-1)); 

    //let's make sure it's not NaN, in which case let's set it equal to -Inf
    if(!arma::is_finite(sub_tree((4+K)*d))){
      sub_tree((4+K)*d) = -arma::datum::inf;
    }
    
    //let's check if there is a divergent transition
    if( (-sub_tree((4+K)*d) - H0) > 1000){
      //add the divergent transition to the global matrix
      sub_tree.subvec(idx,idx+d-1) -= eps * M_inv * sub_tree.subvec(idx + d,idx +2*d-1);
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //update the value of the virial exchange rate
      sub_tree((4+K)*d+4) = (sub_tree((4+K)*d+4) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)))/eps;
      
      //calculate the sign of the virial exchange rate
      sub_tree((4+K)*d+5) = segno(sub_tree((4+K)*d+4));
      
      //calculate the logarithm of the virial exchange rate
      //multiplied by the multinomial weight
      sub_tree((4+K)*d+4) = std::log(std::abs(sub_tree((4+K)*d+4))) + sub_tree((4+K)*d);

    }
    
    //compute the metropolis acceptance rate, which has the mere purpose of tunin
    sub_tree((4+K)*d+2) = std::min(1.0,std::exp(H0+sub_tree((4+K)*d)));

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((4+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,H0,d,log_tau,M_inv,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

      }
      
      //update the value of theta_prop based on the ratio
      //between the 2 weights of the 2 subtrees
      if(!sub_tree((4+K)*d+1)){
        double alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //modify the first, biased sampling
        if(arma::randu() < alpha){
          sub_tree.subvec(4*d,5*d - 1) = sub_tree2.subvec(4*d ,5*d - 1);
        }
        
        //cumulate the log multinomial weights
        sub_tree((4+K)*d) = arma::log_add_exp(sub_tree((4+K)*d),sub_tree2((4+K)*d));
        
        //recalculate the probability for uniform sampling
        alpha = std::exp(sub_tree2((4+K)*d) - sub_tree((4+K)*d));
        
        //uniform sampling
        for(unsigned int i = 1; i<K;i++){
          if(arma::randu() < alpha){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //check the condition of the virial:
      
      //then, cumulates the virial
      add_sign_log_sum_exp(sub_tree( (4+K)*d+4),
                           sub_tree( (4+K)*d+5),
                           sub_tree2((4+K)*d+4),
                           sub_tree2((4+K)*d+5));
      
      //next, check the termination condition       
      sub_tree((4+K)*d+1) += 
        (sub_tree((4+K)*d+4) - sub_tree((4+K)*d) - log(1+sub_tree((4+K)*d+3)) ) < log_tau;
      
      //cumulates metropolis acceptance rates and number of leaves
      sub_tree.subvec((4+K)*d+1,(4+K)*d+3) += sub_tree2.subvec((4+K)*d+1,(4+K)*d+3);

    }
    
  }
  //return the tree
  return sub_tree;
}

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
                     const double& log_tau){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree(5*d + 5) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //compute the value of the new potential energy
    double U = sub_tree(4*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);

    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree(4*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }

    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(4*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree(5*d + 4) = segno(eps);

      //update the extreme value of U on the trajectory
      sub_tree(4*d + 1 + segno(eps)) = U;

      //update the virial exchange rate
      sub_tree(5*d + 5) = ( sub_tree(5*d + 5) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)) )/eps;
      
    }
    
    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(5*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(4*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree(4*d + 2) = sub_tree2(4*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
 
        //update the U value of the left endpoint
        sub_tree(4*d) = sub_tree2(4*d);
      }
      
      //accumulate the rest: metropolis acceptance rates and number of leaves
      
      sub_tree(4*d+1) += sub_tree2(4*d+1);
      sub_tree.subvec(4*d+3,5*d+3) += sub_tree2.subvec(4*d+3,5*d+3);
      
      //then, cumulates the virial
      sub_tree(5*d + 5) += sub_tree2(5*d + 5);
      
      sub_tree(4*d+1) += 
        (sub_tree(5*d+5) / (1+sub_tree(5*d+3)) / sub_tree(5*d+3) ) < log_tau;

      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      sub_tree(5*d+4) = sub_tree2(5*d+4);
      
    }
    
  }
  //return the tree
  return sub_tree;
}

// diagonal matrix case without recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const arma::vec& M_inv){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;
    
    //initialize the value of the virial
    sub_tree(5*d + 5) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));

    //compute the value of the new potential energy
    double U = sub_tree(4*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);
    
    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv(j-idx);

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree(4*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if(  M_inv(j-idx) * std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv(j-idx);
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }

    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree(4*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree(5*d + 4) = segno(eps);

      //update the extreme value of U on the trajectory
      sub_tree(4*d + 1 + segno(eps)) = U;

      //update the virial exchange rate
      sub_tree(5*d + 5) = ( sub_tree(5*d + 5) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)) )/eps;
      
    }

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree(5*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau,M_inv);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree(4*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau,M_inv);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree(4*d + 2) = sub_tree2(4*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the left endpoint
        sub_tree(4*d) = sub_tree2(4*d);
      }
      
      //accumulate the rest: metropolis acceptance rates and number of leaves
      
      sub_tree(4*d+1) += sub_tree2(4*d+1);
      sub_tree.subvec(4*d+3,5*d+3) += sub_tree2.subvec(4*d+3,5*d+3);
      
      //then, cumulates the virial
      sub_tree(5*d + 5) += sub_tree2(5*d + 5);
      
      sub_tree(4*d+1) += 
        (sub_tree(5*d+5) / (1+sub_tree(5*d+3)) / sub_tree(5*d+3) ) < log_tau;

      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      
      sub_tree(5*d+4) = sub_tree2(5*d+4);

    }
    
  }
  //return the tree
  return sub_tree;
}

/* -------------------------- RECYCLED VERSION ------------------------------ */


// identity matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //initialize the value of the virial
    sub_tree((5+K)*d + 5) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));
    
    //compute the value of the new potential energy
    double U = sub_tree((4+K)*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);

    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);
      
      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j));

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree((4+K)*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U;
        U += delta_U;
        
      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }

    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){

      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree((5+K)*d + 4) = segno(eps);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //update the extreme value of U on the trajectory
      sub_tree((5+K)*d + 1 + segno(eps)) = U;
      
      //update the virial exchange rate
      sub_tree((5+K)*d + 5) = ( sub_tree((5+K)*d + 5) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)) )/eps;

    }
    
    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((5+K)*d+3) = 1;

  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree((4+K)*d + 2) = sub_tree2((4+K)*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the left endpoint
        sub_tree((4+K)*d) = sub_tree2((4+K)*d);
      }
      
      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree((4+K)*d+1) += sub_tree2((4+K)*d+1);
      sub_tree.subvec((4+K)*d+3,(5+K)*d+3) += sub_tree2.subvec((4+K)*d+3,(5+K)*d+3);
      
      //then, cumulates the virial
      sub_tree((5+K)*d + 5) += sub_tree2((5+K)*d + 5);
      
      //next, check the termination condition
      sub_tree((4+K)*d+1) += 
        (sub_tree((5+K)*d+5) / (1+sub_tree((5+K)*d+3)) / sub_tree((5+K)*d+3) ) < log_tau;

      //update the value of theta_prop based on the ratio
      //this time we do uniform sampling without bias
      //otherwise we would always have only the extreme value recycled
      if(!sub_tree((5+K)*d+1)){
        for(unsigned int i = 0; i<K;i++){
          if(arma::randu() < 0.5){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }

      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      sub_tree((5+K)*d+4) = sub_tree2((5+K)*d+4);
      
    }
    
  }
  //return the tree
  return sub_tree;
}

// diagonal matrix case with recycling
arma::vec build_tree(arma::vec sub_tree,
                     const Rcpp::Function& nlp,
                     const Rcpp::List& args,
                     const double& eps,
                     const unsigned int depth,
                     const unsigned int& d,
                     arma::uvec& idx_disc,
                     const double& log_tau,
                     const arma::vec& M_inv,
                     const unsigned int& K){
  
  //distinguish the case in which the tree is at depth 0 or more
  if(depth == 0){

    //should we take the step forward or backward?
    //if forward then the position and moment to be modified are those
    //in position from 2*d to 4*d - 1, otherwise from 0 to 2*d-1
    unsigned int idx = (1 + segno(eps)) * d;

    //initialize the value of the virial
    sub_tree((5+K)*d + 5) = -arma::dot(sub_tree.subvec(idx,idx+d-1),sub_tree.subvec(idx + d,idx +2*d-1));
    
    //compute the value of the new potential energy
    double U = sub_tree((4+K)*d + 1 + segno(eps));

    //initialization of the old value and the potential difference
    double theta_old;
    double delta_U;
    
    //permute the order of the discrete parameters
    idx_disc = arma::shuffle(idx_disc);

    unsigned int j;
    //loop for every discontinuous component
    for(unsigned int i = 0; i < d; i++){
      
      //set the current index
      j = idx + idx_disc(i);

      //modify the discrete parameter
      theta_old = sub_tree(j);

      sub_tree(j) = theta_old + eps * segno(sub_tree(d+j)) * M_inv(j-idx);

      //calculation of the difference in potential energy
      delta_U = Rcpp::as<double>(nlp(sub_tree.subvec(idx,idx+d-1),args,true)) - U;

      //calculation of the Metropolis acceptance rate
      sub_tree((4+K)*d+3+j-idx) = std::min(1.0,std::exp(-delta_U));

      //refraction or reflection?
      if( M_inv(j-idx) * std::abs(sub_tree(d+j)) > delta_U ){
        
        //refraction
        sub_tree(d+j) -= segno(sub_tree(d+j)) * delta_U / M_inv(j-idx);
        U += delta_U;

      }else{
        
        //reflection
        sub_tree(j) = theta_old;
        sub_tree(d+j) *= -1.0;

      }
      
    }
    
    //let's check if there is a divergent transition
    if( !arma::is_finite(U)){
      
      //add the divergent transition to the global matrix
      add_div_trans(sub_tree.subvec(idx,idx+d-1));
      
      //communicate that a stopping criterion has been met
      sub_tree((4+K)*d+1) = 1.0;
    }else{
      //if there is no divergent transition we also copy the other values
      
      //since we are at the deepest level of the tree
      //set the left extremes equal to the right ones or vice versa
      sub_tree.subvec(2*d - idx,4*d-1 - idx) = sub_tree.subvec(idx, idx + 2*d -1);

      //confirm that in the current direction has not been refused
      sub_tree((5+K)*d + 4) = segno(eps);

      //also set the value proposed by this leaf equal to the step taken
      //for each recycled sample
      for(unsigned int i = 0; i < K; i++){
        sub_tree.subvec((4+i)*d,(5+i)*d-1) = sub_tree.subvec(0,d-1);
      }

      //update the extreme value of U on the trajectory
      sub_tree((5+K)*d + 1 + segno(eps)) = U;
      
      //update the virial exchange rate
      sub_tree((5+K)*d + 5) = ( sub_tree((5+K)*d + 5) + arma::dot(sub_tree.subvec(0,d-1),sub_tree.subvec(d,2*d-1)) )/eps;

    }

    //initialize the count of the number of integrations made (leaves of the tree)
    sub_tree((5+K)*d+3) = 1;
    
  }else{
    //case where we need to do recursion
    //first build the tree adjacent to the point from which to start the doubling
    
    sub_tree = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau,M_inv,K);

    //if no divergences has been encountered, continue doubling on the other side too
    //otherwise return only this side
    if(!sub_tree((4+K)*d + 1)){
      
      //define the doubled tree
      arma::vec sub_tree2 = build_tree(sub_tree,nlp,args,eps,depth-1,d,idx_disc,log_tau,M_inv,K);

      if(eps > 0){
        //modify the left tree, updating the right limits
        sub_tree.subvec(2*d,4*d-1) = sub_tree2.subvec(2*d,4*d-1);

        //update the U value of the right end
        sub_tree((4+K)*d + 2) = sub_tree2((4+K)*d + 2);
        
      }else{
        //modify the right tree, updating the left limits
        sub_tree.subvec(0,2*d-1) = sub_tree2.subvec(2*d,4*d-1);
        
        //update the U value of the left endpoint
        sub_tree((4+K)*d) = sub_tree2((4+K)*d);
      }
      
      //accumulate the rest: metropolis acceptance rates and number of leaves
      sub_tree((4+K)*d+1) += sub_tree2((4+K)*d+1);
      sub_tree.subvec((4+K)*d+3,(5+K)*d+3) += sub_tree2.subvec((4+K)*d+3,(5+K)*d+3);
      
      //then, cumulates the virial
      sub_tree((5+K)*d + 5) += sub_tree2((5+K)*d + 5);
      
      //next, check the termination criterion
      sub_tree((4+K)*d+1) += 
        (sub_tree((5+K)*d+5) / (1+sub_tree((5+K)*d+3)) / sub_tree((5+K)*d+3) ) < log_tau;

      //update the value of theta_prop based on the ratio
      //this time we do uniform sampling without bias
      //otherwise we would always have only the extreme value recycled
      if(!sub_tree((5+K)*d+1)){
        for(unsigned int i = 0; i<K;i++){
          if(arma::randu() < 0.5){
            sub_tree.subvec((4+i)*d,(5+i)*d - 1) = sub_tree2.subvec((4+i)*d ,(5+i)*d - 1);
          }
        }
      }
      
      //in this case it always takes the extremes of the trajectory
      //so update the direction of the trajectory
      sub_tree((5+K)*d+4) = sub_tree2((5+K)*d+4);

    }
    
  }
  //return the tree
  return sub_tree;
}
