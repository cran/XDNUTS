#include <iostream>
#include <RcppArmadillo.h>
#include <cmath>
#include "globals.h"
#include "globals_interact.h"
#include "globals_functions.h"
#include "recursive_tree.h"
#include "leapfrog.h"
#include "single_nuts.h"
#include "single_hmc.h"
#include "mcmc.h"
#include "epsilon_init.h"
#include "epsilon_adapt.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR


// FUNCTION THAT SET THE XDNUTS SPECIFICATIONS
//' Function that regulates the specifications of the \link{xdnuts} function.
//' @param N_init1 an integer that regulates the number of samples used to
//' adapt the step size.
//' @param N_adapt an integer that regulates the number of samples used to
//' estimate the Mass Matrix with fixed step size.
//' @param N_init2 an integer that regulates the number of samples used to
//'  adapt the step size after the estimation of the Mass Matrix.
//' @param burn_adapt_ratio a numeric scalar \eqn{\in (0,1]} indicating the ratio of warm-up
//' samples to discard in order to estimate the covariance matrix of the parameters.
//' @param keep_warm_up a logical scalar that determines whether the warm-up samples should be returned.
//' @param recycle_only_init a logical value which disables the recycling of the
//' samples from each trajectory once the warm-up phase has terminated.
//' @param max_treedepth an integer that regulates the maximum depth of
//' the binary tree used to approximate Hamilton equation for the exploration
//' of each energy level set of the phase space.
//' @param max_treedepth_init an integer that controls the maximum depth of
//' the binary tree during the step-size adaptation phase. Setting a smaller value 
//' can help avoid wasting valuable time on low-probability areas with suboptimal 
//' algorithm parameters.
//' @param eps_jitter a numeric scalar which regulates the amount of jittering
//' used to perturb the value of the step size for each iteration of the chain
//' after the warm-up phase.
//' @param L_jitter an integer scalar that regulates the amount of jittering used to perturb the
//'  value of the trajectory length if this is specified to be constant.
//'  This occurs when the classic Hamiltonian Monte Carlo algorithm is used through the 
//'  \code{method = "HMC"} option in the \link{xdnuts} function. If \code{L_jitter} \eqn{\geq 1}
//'  each trajectory length is sampled uniformly inside the interval [L - L_jitter , L + L_jitter].
//' @param gamma a numeric value that, in the Nesterov Dual Averaging algorithm, regulates 
//'  the sensitivity of the step size updating scheme to fluctuations in the estimate of the 
//'  mean Metropolis acceptance probability.
//' @param kappa a numeric value that regulates the vanishing of Nesterov Dual Averaging
//' algorithm for the estimation of the step size.
//' @param delta a vector containing the Metropolis acceptance probabilities, 
//'  including both the global and those related to potential differences. Default values are (0.8,0.6).
//'  If the second element of the vector is set to \code{NA}, then the step size calibration is conducted 
//'  solely through the global acceptance probabilities.
//' @param t0 an integer value that makes Nesterov Dual Averaging
//' algorithm for the estimation of the step size less sensitive to early iterations.
//' @param M_type a character value specifying the type of Mass Matrix to estimate:\itemize{
//' \item{\code{"identity"} no Mass Matrix estimation is done.}
//' \item{\code{"diagonal"} a diagonal Mass Matrix is estimated during the warm-up phase.}
//' \item{\code{"dense"} a full dense Mass Matrix is estimated during the warm-up phase.}
//' }
//' @param refresh a numeric scalar bounded in \eqn{(0,1)} which regulates the update frequency of
//' the displayed sampling process state. Default values is 0.1, meaning every 10\% of the total samples.
//' @param l_eps_init a numeric scalar containing the logarithm of the initial value for the step size
//' used to approximate Hamilton differential equation for phase space exploration.
//' @param different_stepsize a boolean value indicating where the adaptation scheme should adapt different step size. 
//'  If \code{TRUE}, a global step size is adapted via Nesterov Dual Averaging algorithm. 
//'  At the same time, for each empirical reflection rate of each component treated as discontinuous the same
//'  algorithm is exploited and the difference between these is obtained through the updating of the discontinuous
//'  components Mass Matrix. Default value is \code{FALSE}.
//' @param mu a numeric scalar containing the value to which the step size is shrunken during the warm-up phase.
//' @param M_cont a vector of length-\eqn{d-k} if \code{M_type = "diagonal"} or a \eqn{(d-k) \times (d-k)} matrix
//' if \code{M_type = "dense"} containing an initial estimate for the Mass Matrix
//' (the inverse of the parameters covariance matrix).
//' If you want to keep it fixed, they should specify \code{N_adapt = 0}.
//' @param M_disc a vector of length-\eqn{k} if \code{M_type = "diagonal"} or 
//' \code{M_type = "dense"} containing an initial estimate for the Mass Matrix
//' (the inverse of the parameters covariances).
//' If one wants to keep it fixed, they should specify \code{N_adapt = 0}.
//' @return an object of class \code{control_xdnuts} containing a named list with all the above parameters.
//'
//' @export set_parameters
// [[Rcpp::export]]
Rcpp::List set_parameters(const unsigned int N_init1 = 50,
                          const unsigned int N_adapt = 200,
                          const unsigned int N_init2 = 75,
                          const double burn_adapt_ratio = 0.1,
                          const bool keep_warm_up = false,
                          const bool recycle_only_init = true,
                          const unsigned int max_treedepth = 10,
                          const unsigned int max_treedepth_init = 10,
                          const double eps_jitter = 0.1,
                          const unsigned int L_jitter = 3,
                          const double gamma = 0.05,
                          const double kappa = 0.75,
                          const Rcpp::Nullable<Rcpp::NumericVector> delta = R_NilValue,
                          const unsigned int t0 = 10,
                          const std::string M_type = "dense",
                          const double refresh = 0.1,
                          const double l_eps_init = NA_REAL,
                          const bool different_stepsize = false,
                          const double mu = NA_REAL,
                          const Rcpp::RObject M_cont = R_NilValue,
                          const Rcpp::RObject M_disc = R_NilValue){
  
  //set delta default value
  arma::vec delta_val;
  if(delta.isNull()){
    delta_val = arma::vec({0.8,0.6});
  }else{
    Rcpp::NumericVector tmp(delta);
    if(tmp.length() < 2){
      Rcpp::stop("'delta' must be a vector of length at least two!");
    }
    delta_val = arma::vec(tmp.begin(),tmp.size(),false);
  }
  
  //check that each input value is admissible
  if(burn_adapt_ratio < 0 || burn_adapt_ratio >= 1){
    Rcpp::stop("'burn_adapt_ratio' should be a scalar in the inteval [0,1)!");
  }
  if(eps_jitter < 0){
    Rcpp::stop("'eps_jitter' must be a positive scalar!");
  }
  if(gamma <= 0){
    Rcpp::stop("'gamma' must be a scalar greater than zero!");
  }
  if(kappa <= 0.5 || kappa > 1){
    Rcpp::stop("'kappa' must be a scalar in the interval (0.5,1]!");
  }
  if(arma::any(delta_val <= 0) || arma::any(delta_val >= 1)){
    Rcpp::stop("'delta' must contain scalars in the interval (0,1)!");
  }
  
  if(std::isnan(delta_val(1)) && different_stepsize ){
    Rcpp::stop(" The second element of 'delta' can't be NA if 'different_stepsize' is set to TRUE!");
  }

  if(M_type != "identity" && M_type != "diagonal" && M_type != "dense"){
    Rcpp::stop("'M_type' must be a character scalar with possible value in \"identity\", \"diagonal\" or \"dense\"");
  }
  if(refresh < 0 || refresh > 1){
    Rcpp::stop("'refresh' must be a scalar in the interval (0,1)!");
  }
  
  if(!M_cont.isNULL() && !M_disc.isNULL()){
    if(M_type == "diagonal" && Rf_isMatrix(M_cont)){
      Rcpp::stop("if 'M_type' = \"diagonal\", M must be a vector containing the diagonal elements!");
    }
    if(M_type == "dense" && !Rf_isMatrix(M_cont)){
      Rcpp::stop("if 'M_type' = \"dense\", M must be a matrix!");
    }
  }
  
  //create the output list
  Rcpp::List control = Rcpp::List::create(Rcpp::Named("N_init1") = N_init1,
                                          Rcpp::Named("N_adapt") = N_adapt,
                                          Rcpp::Named("N_init2") = N_init2,
                                          Rcpp::Named("burn_adapt_ratio") = burn_adapt_ratio,
                                          Rcpp::Named("keep_warm_up") = keep_warm_up,
                                          Rcpp::Named("recycle_only_init") = recycle_only_init,
                                          Rcpp::Named("max_treedepth") = max_treedepth,
                                          Rcpp::Named("max_treedepth_init") = max_treedepth_init,
                                          Rcpp::Named("eps_jitter") = eps_jitter,
                                          Rcpp::Named("L_jitter") = L_jitter,
                                          Rcpp::Named("gamma") = gamma,
                                          Rcpp::Named("kappa") = kappa,
                                          Rcpp::Named("delta") = delta_val,
                                          Rcpp::Named("t0") = t0,
                                          Rcpp::Named("M_type") = M_type,
                                          Rcpp::Named("refresh") = refresh,
                                          Rcpp::Named("l_eps_init") = l_eps_init,
                                          Rcpp::Named("different_stepsize") = different_stepsize,
                                          Rcpp::Named("mu") = mu,
                                          Rcpp::Named("M_cont") = M_cont,
                                          Rcpp::Named("M_disc") = M_disc);
  
  //assign the list an identifying attribute
  control.attr("class") = "control_xdnuts";
  
  //return the list
  return control;
}


// FUNCTION THAT COMPUTE A SINGLE MARKOV CHAIN
//' Function to generate a Markov chain for both continuous and discontinuous posterior distributions.
//' @description The function allows to generate a single Markov Chain for sampling from both continuous and discontinuous
//'  posterior distributions using a plethora of algorithms. Classic Hamiltonian Monte Carlo \insertCite{duane1987hybrid}{XDNUTS} ,
//'   NUTS \insertCite{hoffman2014no}{XDNUTS} and XHMC \insertCite{betancourt2016identifying}{XDNUTS} are embedded into the framework
//' described in \insertCite{nishimura2020discontinuous}{XDNUTS}, which allows to deal with such posteriors.
//' Furthermore, for each method, it is possible to recycle samples from the trajectories using
//' the method proposed by \insertCite{Nishimura_2020}{XDNUTS}.
//' This is used to improve the estimation of the mass matrix during the warm-up phase
//' without requiring significant additional computational costs.
//' This function should not be used directly, but only through the user interface provided by \link{xdnuts}.
//' @param theta0 a vector of length-\eqn{d} containing the starting point of the chain.
//' @param nlp a function object that takes three arguments:
//' \describe{\item{par}{a vector of length-\eqn{d} containing the value of the parameters.}
//' \item{args}{a list object that contains the necessary arguments, namely data and hyperparameters.}
//' \item{eval_nlp}{a boolean value, \code{TRUE} for evaluating only the model\'s negative log posterior, 
//' \code{FALSE} to evaluate the gradient with respect to the continuous components of the posterior.}}
//' @param args the necessary arguments to evaluate the negative log posterior and its gradient.
//' @param k the number of parameters that induce a discontinuity in the posterior distribution.
//' @param N integer containing the number of post warm-up samples to evaluate.
//' @param K integer containing the number of recycled samples from each trajectory during the warm-up phase or beyond.
//' @param tau the threshold for the exhaustion termination criterion described in \insertCite{betancourt2016identifying}{XDNUTS}.
//' @param L the desired length of the trajectory of classic Hamiltonian Monte Carlo algorithm.
//' @param thin integer containing the number of samples to discard in order to produce a final iteration of the chain.
//' @param chain_id the identification number of the chain.
//' @param verbose a boolean value that controls whether to print all the information regarding the sampling process.
//' @param control an object of class \code{control_xdnuts} containing the specifications for the algorithm.
//' See the \link{set_parameters} function for detail.
//' 
//' @return a named list containing: \describe{
//'\item{values}{a \eqn{N \times d} matrix containing the sample from the
//'target distribution (if convergence has been reached).}
//'\item{energy}{a vector of length-\eqn{N} containing the Markov Chain of the energy level sets.}
//'\item{delta_energy}{a vector of length-\eqn{N} containing the Markov Chain of the first difference energy level sets.}
//'\item{step_size}{a vector of length-\eqn{N} containing the sampled step size used for each iteration.}
//'\item{step_length}{a vector of length-\eqn{N} containing the length of each trajectory of the chain.}
//'\item{alpha}{a vector of length-\eqn{k + 1} containing the estimate of the Metropolis acceptance probabilities.
//' The first element of the vector is the estimated global acceptance probability. The remaining k elements are the 
//' estimate rate of reflection for each parameters which travels coordinate-wise through some discontinuity.}
//' \item{warm_up}{a \eqn{N_{adapt} \times d} matrix containing the sample of the chain
//' coming from the warm-up phase. If \code{keep_warm_up = FALSE} inside the \code{control}
//' argument, nothing is returned.}
//' \item{div_trans}{a \eqn{M \times d} matrix containing the locations where a divergence has been 
//' encountered during the integration of Hamilton equation. Hopefully \eqn{M \ll N}, and even better if \eqn{M = 0}.}
//' \item{M_cont}{the Mass Matrix of the continuous components estimated during the warm-up phase.
//'               Based on the \code{M_type} value of the \code{control} arguments, this could be an empty object, a vector or a matrix.}
//' \item{M_disc}{the Mass Matrix of the discontinuous components estimated during the warm-up phase.
//'               Based on the \code{M_type} value of the \code{control} arguments, this could be an empty object or a vector.}}
//'
//' @references
//'  \insertAllCited{}
//' 
//' @export main_function
// [[Rcpp::export]]
Rcpp::List main_function(const arma::vec& theta0,
                         const Rcpp::Function& nlp,
                         const Rcpp::List& args,
                         const unsigned int k,
                         const unsigned int N,
                         unsigned int K,
                         double tau,
                         unsigned int L,
                         int thin,
                         const unsigned int& chain_id,
                         const bool verbose,
                         const Rcpp::List& control){
  
  //get parameter dimension
  unsigned int d = theta0.size();
  
  //report to console the type of algorithm used
  
  //no discontinuous components
  if(k == 0 && verbose){
    if(tau == 0 && L == 0){
      Rcpp::Rcout << "\nNo discontinuous parameters found, sampling with classic NUTS algorithm..." << std::endl;
    }else if(L == 0){
      Rcpp::Rcout << "\nNo discontinuous parameters found, sampling with classic XHMC algorithm..." << std::endl;
    }else if(tau == 0){
      Rcpp::Rcout << "\nNo discontinuous parameters found, sampling with classic HMC algorithm..." << std::endl;
    }
  }
  
  //only discontinuous components
  if(k == d && verbose){
    if(tau == 0 && L == 0){
      Rcpp::Rcout << "\nNo continuous parameters found, sampling with pure DNUTS algorithm..." << std::endl;
    }else if(L == 0){
      Rcpp::Rcout << "\nNo continuous parameters found, sampling with pure DXHMC algorithm..." << std::endl;
    }else if(tau == 0){
      Rcpp::Rcout << "\nNo continuous parameters found, sampling with pure DHMC algorithm..." << std::endl;
    }
  }
  
  //set the initial value of the chain
  arma::vec theta = theta0; 
  
  //get desired Mass Matrix type
  std::string M_type = Rcpp::as<std::string>(control["M_type"]);
  
  //initialize all possible Mass Matrix:
  
  //diagonal
  arma::vec M_cont_diag,M_disc,M_inv_cont_diag,M_inv_disc; 
  
  //dense
  arma::mat M_cont_dense, M_inv_cont_dense;
  
  //initialize them if provided through the set_parameter() function
  if(!Rf_isNull(control["M_cont"]) && !Rf_isNull(control["M_disc"]) && M_type != "identity"){
    MM(M_cont_diag,M_disc,M_inv_cont_diag,M_inv_disc,M_cont_dense,M_inv_cont_dense,control,M_type);
    
  }else if(Rcpp::as<bool>(control["different_stepsize"]) && k != 0){
    //set the diagonal elements to one if the adapting scheme has to be done with different stepsize
    //and no Mass Matrix is given in input
    M_disc = arma::ones<arma::vec>(k);
    M_inv_disc = arma::ones<arma::vec>(k);
    M_cont_diag = arma::ones<arma::vec>(d-k);
    M_inv_cont_diag = arma::ones<arma::vec>(d-k);
    M_cont_dense = arma::eye(d-k,d-k);
    M_inv_cont_dense = arma::eye(d-k,d-k);
    
    //if M_type is identity change it to diagonal and throw an allert
    control["M_type"] = "diagonal";
    if(verbose){
      Rcpp::Rcout << " 'M_type' set to 'diagonal' for compatibility with 'differnt_stepsize = TRUE'. " << std::endl;
    }
  }
  
  //initialize the empy global matrix for divergent transition location storing 
  create_DT(d);
  
  // get warm-up sample sizes
  unsigned int N_init1 = Rcpp::as<unsigned int>(control["N_init1"]);
  unsigned int N_adapt = Rcpp::as<unsigned int>(control["N_adapt"]);
  unsigned int N_init2 = Rcpp::as<unsigned int>(control["N_init2"]);
  
  //get maximum tree depth
  unsigned int max_treedepth;
  if(L == 0){
    //if NUTS/XHMC is used, get it from the control argument
    max_treedepth = Rcpp::as<unsigned int>(control["max_treedepth"]);
  }else{
    //if HMC is used, set it as the L argument
    max_treedepth = L;
    control["max_treedepth"] = L;
  }
  
  //get the logarithm of tau, threshold of the virial termination criterion
  double log_tau;
  if(tau != 0){
    //if XHMC
    log_tau = std::log(tau);
  }else{
    //if NUTS/HMC set it to one thousend
    log_tau = 1000;
  }
  
  //get console update frequency
  double refresh = Rcpp::as<double>(control["refresh"]);
  
  //get the burn-in ratio of the warm-up phase
  double bar = Rcpp::as<double>(control["burn_adapt_ratio"]);
  
  //compute discontinuous components indexes
  arma::uvec idx_disc(k);
  for(unsigned int i=0; i< k; i++){
    idx_disc(i) = d-k+i;
  }
  
  //get initial step-size
  double eps = std::exp(Rcpp::as<double>(control["l_eps_init"]));
  
  //if verbose, print the current step size
  if(verbose){
    Rcpp::Rcout << "Chain " << chain_id << ", current step-size: " << eps << std::endl;
  }
  
  //first step size calibration
  if(N_init1 > 0){
    
    Rcpp::Rcout << "Chain " << chain_id << ", Step size first calibration..." << std::endl;
    
    //adaptive procedure
    adapt_stepsize_wrapper(theta,
                           eps,
                           nlp,
                           args,
                           d,
                           k,
                           idx_disc,
                           N_init1,
                           control,
                           M_cont_diag,
                           M_disc,
                           M_inv_cont_diag,
                           M_inv_disc,
                           M_cont_dense,
                           M_inv_cont_dense,
                           M_type,
                           log_tau,
                           L,
                           verbose,
                           chain_id);
    
    //if verbose, print the new step size value
    if(verbose){
      Rcpp::Rcout << "Chain " << chain_id << ", current step-size: " << eps << std::endl;
    }
  }
  
  //if the step size is not admissible throw an error
  if(std::isnan(eps)){
    Rcpp::stop("step-size value not specified or the adaptive scheme run into an error!");
  }
  
  //set the number of recycled samples for each iteration of the warm up phase
  //if no mass matrix estimation is done, there is no reason to recycle
  unsigned int K_adapt;
  if(M_type == "identity"){
    K_adapt = 1;
  }else{
    K_adapt = K;
  }
  
  //initialize warm-up samples matrix
  arma::mat warm_up(N_adapt*K_adapt,d);
  
  //generate the step size for each warm up iteration
  arma::vec step_size_warm = eps *
    ( ( (1-Rcpp::as<double>(control["eps_jitter"]))) + arma::randu(N_adapt) *
    (2.0 * Rcpp::as<double>(control["eps_jitter"])) );
  
  //initialize the trajectory length vector
  arma::uvec step_length(N);
  
  //get the amount of desired jittering for the trajectory length
  //specific only for classic HMC
  unsigned int L_jitter = Rcpp::as<unsigned int>(control["L_jitter"]);
  
  //initialize the energy vector
  arma::vec energy(N);
  
  //initialize first difference energy vector
  arma::vec delta_energy(N);
  
  //initialize empirical Metropolis acceptance probability vector
  arma::vec alpha(k+1*(d != k));
  
  //WARM UP PHASE
  if(N_adapt > 0){
    
    //mcmc
    mcmc_wrapper(warm_up,
                 step_size_warm,
                 step_length,
                 energy,
                 delta_energy,
                 alpha,
                 max_treedepth,
                 refresh,
                 theta,
                 nlp,
                 args,
                 N_adapt,
                 bar,
                 d,
                 k,
                 idx_disc,
                 M_cont_diag,
                 M_disc,
                 M_inv_cont_diag,
                 M_inv_disc,
                 M_cont_dense,
                 M_inv_cont_dense,
                 K_adapt,
                 M_type,
                 true,
                 chain_id,
                 thin,
                 log_tau,
                 L,
                 L_jitter);
    
  }
  
  //second step size calibration
  //if(Rcpp::as<std::string>(control["M_type"]) != "identity" && N_init2 > 0 && N_adapt > 0){
  if(N_init2 > 0){
    Rcpp::Rcout << "Chain " << chain_id << ", Step size second calibration..." << std::endl;
    
    //adaptive procedure
    adapt_stepsize_wrapper(theta,
                           eps,
                           nlp,
                           args,
                           d,
                           k,
                           idx_disc,
                           N_init2,
                           control,
                           M_cont_diag,
                           M_disc,
                           M_inv_cont_diag,
                           M_inv_disc,
                           M_cont_dense,
                           M_inv_cont_dense,
                           M_type,
                           log_tau,
                           L,
                           verbose,
                           chain_id);
    
    //if verbose, print step size new value
    if(verbose){
      Rcpp::Rcout << "Chain " << chain_id << ", current step-size: " << eps << std::endl;
    }
    
  }
  
  //check the admissibility of the new step size
  if(std::isnan(eps)){
    Rcpp::stop("step-size value not specified or the adaptive scheme run into an error!");
  }
  
  // SAMPLING PHASE
  
  //do we want to recycle the samples even now?
  if(Rcpp::as<bool>(control["recycle_only_init"]) == true || thin > 1){
    K = 1;
  }
  
  //print to console the mass matrices
  // if(verbose){
  //   Rcpp::Rcout << "M_cont_diag: " << M_cont_diag << std::endl;
  //   Rcpp::Rcout << "M_disc: " << M_disc << std::endl;
  //   Rcpp::Rcout << "M_inv_cont_diag: " << M_inv_cont_diag << std::endl;
  //   Rcpp::Rcout << "M_inv_disc: " << M_inv_disc << std::endl;
  //   Rcpp::Rcout << "M_cont_dense: " << M_cont_dense << std::endl;
  //   Rcpp::Rcout << "M_inv_cont_dense: " << M_inv_cont_dense << std::endl;
  //   
  // }
  
  //initialize the counting of divergent transition and allow storing of their location
  stora();
  
  //initialize the sampling matrix
  arma::mat samples(N*K,d);
  
  //generate the step size foe each sampling iteration
  arma::vec step_size = eps *
    ( ( (1-Rcpp::as<double>(control["eps_jitter"]))) + arma::randu(N) *
    (2.0 * Rcpp::as<double>(control["eps_jitter"])) );
  
  //mcmc
  mcmc_wrapper(samples,
               step_size,
               step_length,
               energy,
               delta_energy,
               alpha,
               max_treedepth,
               refresh,
               theta,
               nlp,
               args,
               N,
               bar,
               d,
               k,
               idx_disc,
               M_cont_diag,
               M_disc,
               M_inv_cont_diag,
               M_inv_disc,
               M_cont_dense,
               M_inv_cont_dense,
               K,
               M_type,
               false,
               chain_id,
               thin,
               log_tau,
               L,
               L_jitter);
  
  //output list building 
  
  Rcpp::List output;
  
  //samples
  output["values"] = samples;
  
  //energy levels
  output["energy"] = energy;
  
  //energy jumps
  output["delta_energy"] = delta_energy;
  
  //step size
  output["step_size"] = step_size;
  
  //step length
  output["step_length"] = step_length;
  
  //empirical estimates of Metropolis acceptance probability
  output["alpha"] = alpha / N;
  
  //warm up samples
  if(Rcpp::as<bool>(control["keep_warm_up"]) && N_adapt > 0){
    output["warm_up"] = warm_up;
  }
  
  //eventual divergent transitions
  if(n_dt > 0){
    output["div_trans"] = get_DT();
  }
  
  //Mass Matrix used
  if(M_type == "diagonal"){
    output["M_cont"] = arma::square(M_cont_diag);
    output["M_disc"] = arma::square(M_disc);
  }else if(M_type == "dense"){
    output["M_cont"] = M_cont_dense * M_cont_dense.t();
    output["M_disc"] = arma::square(M_disc);
  }
  
  //return output
  return output;
  
}

// FUNCTIONS TO BUILD THE TRAJECTORIES
//' Function that approximate the Hamiltonian Flow for given starting values
//' of the position and momentum of a particle in the phase space defined by the 
//' kinetic and potential energy provided in input. 
//' @param theta0 a numeric vector of length \eqn{d} representing
//'  the starting position vector for the particle.
//' @param m0 a numeric vector of length \eqn{d} representing 
//' the starting momenta vector for the particle.
//' @param nlp a function object that evaluate the negative of the logarithm of 
//' a probability density function, and its gradient, i.e. the potential energy function of the system.
//' @param args a list object containing the arguments to be passed to the function \code{nlp}.
//' @param eps a numeric scalar indicating the step size for the \emph{leapfrog} integrator.
//' @param k an integer scalar indicating the number of discontinuous components of \code{theta0}.
//' @param M_cont either a vector or a squared matrix, of the same length/dimension of
//' the position/momenta vector, representing the continuous components mass matrix.
//' @param M_disc a vector of the same length of the position/momenta vector,
//'  representing the discontinuous components mass matrix.
//' @param max_it an integer value indicating the length of the trajectory. 
//' This quantity times \code{eps} is equal to the approximated integration time
//' of the Hamiltonian flow.
//' @return a data frame that summarizes the approximated Hamiltonian flow.
//' \itemize{
//' \item{The first \eqn{d} columns contain the particle position evolution.}
//' \item{The second \eqn{d} columns contain the particle momenta evolution.}
//' \item{The \eqn{2d + 1} column contains the Hamiltonian evolution.}
//' \item{The \eqn{2d + 2} column contains the evolution of the No U-Turn Sampler termination criterion.}
//' \item{The \eqn{2d + 3} column contains the evolution of the virial exhaustion termination criterion.}
//' \item{Acceptance and or refraction probabilities. This depends on the value of \code{k}.}
//' \item{Reflession dummy indicators.}
//' \item{Divergent transition dummy indicators.}
//' } 
//' @export trajectories
// [[Rcpp::export]]
Rcpp::DataFrame trajectories(const Rcpp::NumericVector& theta0,
                             const arma::vec& m0,
                             const Rcpp::Function& nlp,
                             const Rcpp::List& args,
                             const double& eps,
                             const unsigned int& k,
                             const Rcpp::RObject& M_cont,
                             const Rcpp::RObject& M_disc,
                             const unsigned int& max_it){
  
  // get the dimension of the parameter
  unsigned int d = theta0.size();
  
  // get the dimension names
  Rcpp::RObject dim_names0 = theta0.names();
  
  // initialize the position names
  Rcpp::CharacterVector position_names(d);
  
  // initialize the momentum names
  Rcpp::CharacterVector momentum_names(d);
  
  // define position and momenutm names
  if(dim_names0 == R_NilValue){
    for(unsigned int i = 0; i < d; i++){
      position_names[i] = "theta_" + std::to_string(1+i);
      momentum_names[i] = "m_" + std::to_string(1+i);
    }
  }else{
    position_names = theta0.names();
    for(unsigned int i = 0; i < d; i++){
      momentum_names[i] = Rcpp::as<std::string>(position_names[i]) + "'";
    }
  }
  
  // create the index for the discontiuous components
  arma::uvec idx_disc(k);
  for(unsigned int i=0; i< k; i++){
    idx_disc(i) = d-k+i;
  }
  
  // get the dimension of the output matrix
  unsigned int out_dim = 2*d + 5;
  
  if(k == d){
    
    // if we are in the full discontinuous case we must add d values
    // for the refraction alphas and d for the refraction indicator
    out_dim += 2*d;
    
  }else if(k == 0){
    
    // if we are in the full continuous case we must add only one value
    out_dim += 1;
    
  }else{
    
    // if we are in the mixed case we must add one for the global, 
    // and 2*k for the refractions
    
    out_dim += 1 + 2*k;
  }
  
  // initialize the output matrix
  arma::mat out(max_it,out_dim);
  
  // initialize the position
  arma::vec theta = theta0;
  out.row(0).subvec(0,d-1) = theta.t();
  
  // initialize the momentum
  arma::vec m = m0;
  out.row(0).subvec(d,2*d-1) = m.t();
  
  // create a vector for the cumulated momenta
  arma::vec cum_momenta(d);
  
  // initializze the current value of the virial
  double virial = arma::sum(theta % m);
  
  // initialize the current value of the virial first difference
  double delta_virial;
  
  // create the scalar for the log absolute value of the cumulative virial exchange rate
  double log_abs_sum_virial;
  
  // create the sign for the virial
  double delta_virial_sign;
  
  // initialize the cumulate hamiltonian (normalization constant for the virial criterion)
  double sum_H = -arma::datum::inf;
  
  // initialize the hamiltonian
  double H;
  
  // discriminate the variouos cases
  if(Rf_isNull(M_cont) | Rf_isNull(M_disc)){
    
    // identity matrix case
    
    if(d == k){
      
      // initialize the cumulated momenta vector
      cum_momenta = arma::sign(m0);
      
      // compute the potential energy
      double U = Rcpp::as<double>(nlp(theta0,args,true));
      
      // set the initial alphas equal to one
      out.row(0).subvec(2*d+4,3*d+3) = arma::ones<arma::rowvec>(k);
      
      // save the current potential energy
      out(0,2*d) = U;
      
      // compute the hamiltonian for the function
      H = U + arma::sum(arma::abs(m0));
      out.col(2*d+1) = arma::ones<arma::vec>(max_it) * H;
      
      // add the NUTS criterion
      out(0,2*d+2) = 1.0;
      
      // add the virial exhaustion criterion
      //out(0,2*d+2) = arma::sum(m % theta);
      
      // initialize the log_abs_sum del virial and its sign
      //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
      log_abs_sum_virial = -arma::datum::inf;
      //delta_virial_sign = segno(out(0,2*d+2));
      delta_virial_sign = 0.0;
      
      //initialization of the new vector and the potential difference
      double theta_old;
      double delta_U;
      unsigned int j;
      
      // loop for each iteration
      for( unsigned int i = 1; i < max_it; i++){
        
        //permute the order of the discrete parameters
        idx_disc = arma::shuffle(idx_disc);
        
        //loop for every discontinuous component
        for(unsigned int ii = 0; ii < d; ii++){
          
          //set the current index
          j = idx_disc(ii);
          
          //modify the discrete parameter
          theta_old = theta(j);
          
          theta(j) = theta_old + eps * segno(m(j));
          
          //calculation of the difference in potential energy
          delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
          
          //let's make sure it's finite
          if(std::isnan(delta_U)){
            delta_U = -arma::datum::inf;
            theta(j) = theta_old;
            
            //report the divergent transition
            out(i,4*d+4) = 1.0;
            
            break;
            
          }else{
            
            //calculation of the Metropolis acceptance rate
            out(i,2*d+4+j) += std::exp(-delta_U);
            
            //refraction or reflection?
            if( std::abs(m(j)) > delta_U ){
              
              //refraction
              m(j) -= segno(m(j)) * delta_U;
              U += delta_U;
              
            }else{
              
              //reflection
              theta(j) = theta_old;
              m(j) *= -1.0;
              
              // report it
              out(i,3*d+4 + j) = 1.0;
              
            }
          }
        }
        
        // save the evolution of the flow in the phase space
        out.row(i).subvec(0,d-1) = theta.t();
        out.row(i).subvec(d,2*d-1) = m.t();
        
        // add the potential energy of the current system
        out(i,2*d) = U;
        
        //1) compute the NUTS criterion
        
        // cumulate the momenta vector
        cum_momenta += arma::sign(m);
        
        // compute the criterion
        out(i,2*d+2) = arma::sum(cum_momenta % m);
        
        //2) compute the virial criterion
        
        // save the last value of the virial
        delta_virial = virial;
        
        // compute the new virial
        virial = sum(theta % m);
        
        // approximate the exchange rate
        delta_virial = (virial - delta_virial) / eps;
        
        // update the log sum exp of the virial log virial exchange rates
        add_sign_log_sum_exp(log_abs_sum_virial,
                             delta_virial_sign,
                             std::log(std::abs(delta_virial)) - out(i,2*d+1),
                             segno(delta_virial));
        
        // cumulate the Hamiltonians
        sum_H = log_add_exp(sum_H,-out(i,2*d+1));
        
        // save the virial exhaustion criterion
        out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
        
      }
      
    }else if(k == 0){
      
      // initialize the cumulative momenta vector
      cum_momenta = m0;
      
      // set the current acceptance rate
      out(0,2*d+4) = 1.0;
      
      // compute the potential energy of the system
      out(0,2*d) = Rcpp::as<double>(nlp(theta0,args,true));
      
      // compute the hamiltonian for the function
      H = out(0,2*d) + arma::sum(arma::abs(m0));
      out(0,2*d+1) = H;
      
      // add the NUTS criterion
      out(0,2*d+2) = 1.0;
      
      // add the virial exhaustion criterion
      //out(0,2*d+2) = arma::sum(m % theta);
      
      // initialize the log_abs_sum del virial and its sign
      //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
      log_abs_sum_virial = -arma::datum::inf;
      //delta_virial_sign = segno(out(0,2*d+2));
      delta_virial_sign = 0.0;
      
      // loop for each iteration
      for( unsigned int i = 1; i < max_it; i++){
        
        //compute the gradient
        arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
        
        //if the gradient is finite we can continue
        if(grad.is_finite()){
          
          //continuous momentum update by half step size
          m -= 0.5 * eps * grad;
          
          //continuous parameter update by one step size
          theta += eps * m;
          
          //compute the gradient
          grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
          
          //if the gradient is finite we can continue
          if(grad.is_finite()){
            
            //continuous momentum update by half step size
            m -= 0.5 * eps * grad;
            
            // compute the potential energy of the system
            out(i,2*d) = Rcpp::as<double>(nlp(theta,args,true));
            
            //compute the Hamiltonian of the system
            out(i,2*d+1) = out(i,2*d) + 0.5*arma::sum(arma::square(m)); 
            
            //let's make sure it's not NaN, in which case let's set it equal to +Inf
            if(!std::isfinite(out(i, 2*d + 1))){
              out(i,2*d+1) = arma::datum::inf;
            }
            
            //let's check if there is a divergent transition
            if( (out(i,2*d+1) - H) > 1000){
              
              //add the divergent transition to the global matrix
              theta -= 0.5 * eps * m;
              
              //report the divergent transition
              out(i,2*d+5) = 1.0;
              
            }else{
              //report the metropolis acceptance rate
              out(i,2*d+4) = std::exp(H-out(i,2*d+1));
            }
          }else{
            //add the divergent transition to the global matrix
            theta -= 0.5 * eps * m;
            
            //report the divergent transition
            out(i,2*d+5) = 1.0;    
          }
          
        }else{
          //report the divergent transition
          out(i,2*d+5) = 1.0;
        }
        
        // save the evolution of the flow in the phase space
        out.row(i).subvec(0,d-1) = theta.t();
        out.row(i).subvec(d,2*d-1) = m.t();
        
        //1) compute the NUTS criterion
        
        // cumulate the momenta vector
        cum_momenta += m;
        
        // compute the criterion
        out(i,2*d+2) = arma::sum(cum_momenta % m);
        
        //2) compute the virial criterion
        
        // save the last value of the virial
        delta_virial = virial;
        
        // compute the new virial
        virial = sum(theta % m);
        
        // approximate the exchange rate
        delta_virial = (virial - delta_virial) / eps;
        
        // update the log sum exp of the virial log virial exchange rates
        add_sign_log_sum_exp(log_abs_sum_virial,
                             delta_virial_sign,
                             std::log(std::abs(delta_virial)) - out(i,2*d+1),
                             segno(delta_virial));
        
        // cumulate the Hamiltonians
        sum_H = log_add_exp(sum_H,-out(i,2*d+1));
        
        // save the virial exhaustion criterion
        out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
        
      }
      
    }else{
      
      // initialize the cumulative momenta vector
      cum_momenta.subvec(0,d-k-1) = m0.subvec(0,d-k-1);
      cum_momenta.subvec(d-k,d-1) = arma::sign(m0.subvec(d-k,d-1));
      
      // set the current acceptance rate equal to one
      out.row(0).subvec(2*d+4,2*d+4+k) = arma::ones<arma::rowvec>(k+1);
      
      // compute the initial potential energy of the system
      out(0,2*d) = Rcpp::as<double>(nlp(theta0,args,true));
      
      // add the hamiltonian for the function
      H = out(0,2*d) + 
        0.5*arma::sum(arma::square(m0.subvec(0,d-k-1))) + 
        arma::sum(arma::abs(m0.subvec(d-k,d-1)));
      out(0,2*d+1) = H;
      
      // add the NUTS criterion
      out(0,2*d+2) = 1.0;
      
      // add the virial exhaustion criterion
      //out(0,2*d+2) = arma::sum(m % theta);
      
      // initialize the log_abs_sum del virial and its sign
      //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
      log_abs_sum_virial = -arma::datum::inf;
      //delta_virial_sign = segno(out(0,2*d+2));
      delta_virial_sign = 0.0;
      
      double U;
      double theta_old;
      double delta_U;
      unsigned int j;
      
      // loop for each iteration
      for( unsigned int i = 1; i < max_it; i++){
        
        //compute the gradient
        arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
        
        //if the gradient is finite we can continue
        if(grad.is_finite()){
          
          //continuous momentum update by half step size
          m.subvec(0,d-k-1) -= 0.5 * eps * grad;
          
          //continuous parameter update by half step size
          theta.subvec(0,d-k-1) += 0.5 * eps * m.subvec(0,d-k-1);
          
          //compute the value of the new potential energy
          U = Rcpp::as<double>(nlp(theta,args,true));
          
          // if the potential energy is finite then we continue
          if(std::isfinite(U)){
            
            //permute the order of the discrete parameters
            idx_disc = arma::shuffle(idx_disc);
            
            //loop for every discontinuous component
            for(unsigned int ii = 0; ii < k; ii++){
              
              //set the current index
              j = idx_disc(ii);
              
              //modify the discrete parameter
              theta_old = theta(j);
              
              theta(j) = theta_old + eps * segno(m(j));
              
              //calculation of the difference in potential energy
              delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
              
              //calculation of the Metropolis acceptance rate
              out(i,d+5+j+k) = std::exp(-delta_U);
              
              //refraction or reflection?
              if( std::abs(m(j)) > delta_U ){
                
                //refraction
                m(j) -= segno(m(j)) * delta_U;
                U += delta_U;
                
              }else{
                
                //reflection
                theta(j) = theta_old;
                m(j) *= -1.0;
                
                //report it
                out(i,d+5+j+2*k) = std::exp(-delta_U);
                
                
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
              
              // compute the new potential energy
              out(i,2*d) = Rcpp::as<double>(nlp(theta,args,true));
              
              //compute the hamiltonian
              out(i,2*d+1) = out(i,2*d) + 
                0.5*arma::sum(arma::square(m.subvec(0 ,d-k-1))) + 
                arma::sum(arma::abs(m.subvec(d-k,d-1)));
              
              //let's make sure it's not NaN, in which case let's set it equal to +Inf
              if(!std::isfinite(out(i,2*d))){
                out(i,2*d+1) = out(i,2*d) = arma::datum::inf;
              }
              
              //let's check if there is a divergent transition
              if( (out(i,2*d+1) - H) > 1000){
                
                //add the divergent transition to the global matrix
                theta.subvec(0,d-k-1) -= 0.5 * eps * m.subvec(0,d-k-1);
                
                //report the divergent transition
                out(i,2*d+5+2*k) = 1.0;
                
              }else{
                //update the metropolis acceptance rate
                out(i,2*d+4) += std::exp(H-out(i,2*d+1));
              }
            }else{
              
              //add the divergent transition to the global matrix
              theta.subvec(0,d-k-1) -= 0.5 * eps * m.subvec(0,d-k-1);
              
              //report the divergent transition
              out(i,2*d+5+2*k) = 1.0;
            }
          }else{
            
            //add the divergent transition to the global matrix
            theta.subvec(0,d-k-1) -= 0.5 * eps * m.subvec(0,d-k-1);
            
            //report the divergent transition
            out(i,2*d+5+2*k) = 1.0;
          }
        }else{
          //add the divergent transition to the global matrix
          
          //report the divergent transition
          out(i,2*d+5+2*k) = 1.0;
        }
        
        // save the evolution of the flow in the phase space
        out.row(i).subvec(0,d-1) = theta.t();
        out.row(i).subvec(d,2*d-1) = m.t();
        
        //1) compute the NUTS criterion
        
        // cumulate the momenta vector
        cum_momenta.subvec(0,d-k-1) += m.subvec(0,d-k-1);
        cum_momenta.subvec(d-k,d-1) += arma::sign(m.subvec(d-k,d-1));
        
        // compute the criterion
        out(i,2*d+2) = arma::sum(cum_momenta % m);
        
        //2) compute the virial criterion
        
        // save the last value of the virial
        delta_virial = virial;
        
        // compute the new virial
        virial = sum(theta % m);
        
        // approximate the exchange rate
        delta_virial = (virial - delta_virial) / eps;
        
        // update the log sum exp of the virial log virial exchange rates
        add_sign_log_sum_exp(log_abs_sum_virial,
                             delta_virial_sign,
                             std::log(std::abs(delta_virial)) - out(i,2*d+1),
                             segno(delta_virial));
        
        // cumulate the Hamiltonians
        sum_H = log_add_exp(sum_H,-out(i,2*d+1));
        
        // save the virial exhaustion criterion
        out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
        
      }
      
    }
    
    
  }else{
    
    if(!Rf_isMatrix(M_cont)){
      
      // diagonal matrix case
      
      // compute the inverses
      arma::vec M_inv_cont = 1/ Rcpp::as<arma::vec>(M_cont);
      arma::vec M_inv_disc = 1/arma::sqrt(Rcpp::as<arma::vec>(M_disc));
      
      if(d == k){
        
        // initialize the cumulative momenta vector
        cum_momenta = arma::sign(m0);
        
        // set the current acceptance rate equal to one
        out.row(0).subvec(2*d+4,3*d+3) = arma::ones<arma::rowvec>(k);
        
        // compute the potential energy
        double U = Rcpp::as<double>(nlp(theta0,args,true));
        out(0,2*d) = U;
        
        // compute the hamiltonian for the function
        H = U + arma::dot(arma::abs(m0),M_inv_disc);
        out.col(2*d+1) = arma::ones<arma::vec>(max_it) * H;
        
        // add the NUTS criterion
        out(0,2*d+2) = 1.0;
        
        // add the virial exhaustion criterion
        //out(0,2*d+2) = arma::sum(m % theta);
        
        // initialize the log_abs_sum del virial and its sign
        //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
        log_abs_sum_virial = -arma::datum::inf;
        //delta_virial_sign = segno(out(0,2*d+2));
        delta_virial_sign = 0.0;
        
        //initialization of the new vector and the potential difference
        double theta_old;
        double delta_U;
        unsigned int j;
        
        // loop for each iteration
        for( unsigned int i = 1; i < max_it; i++){
          
          //permute the order of the discrete parameters
          idx_disc = arma::shuffle(idx_disc);
          
          //loop for every discontinuous component
          for(unsigned int ii = 0; ii < d; ii++){
            
            //set the current index
            j = idx_disc(ii);
            
            //modify the discrete parameter
            theta_old = theta(j);
            
            theta(j) = theta_old + eps * M_inv_disc(j) * segno(m(j));
            
            //calculation of the difference in potential energy
            delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
            
            //let's make sure it's finite
            if(std::isnan(delta_U)){
              delta_U = -arma::datum::inf;
              theta(j) = theta_old;
              
              //report the divergent transition
              out(i,4*d+4) = 1.0;
              
              break;
              
            }else{
              
              //calculation of the Metropolis acceptance rate
              out(i,2*d+4+j) += std::exp(-delta_U);
              
              //refraction or reflection?
              if( M_inv_disc(j) * std::abs(m(j)) > delta_U ){
                
                //refraction
                m(j) -= segno(m(j)) * delta_U / M_inv_disc(j);
                U += delta_U;
                
              }else{
                
                //reflection
                theta(j) = theta_old;
                m(j) *= -1.0;
                
                // report it
                out(i,3*d+4 + j) = 1.0;
                
              }
            }
          }
          
          // save the evolution of the flow in the phase space
          out.row(i).subvec(0,d-1) = theta.t();
          out.row(i).subvec(d,2*d-1) = m.t();
          
          // save the current valur for the potential energy
          out(i,2*d) = U;
          
          //1) compute the NUTS criterion
          
          // cumulate the momenta vector
          cum_momenta += arma::sign(m);
          
          // compute the criterion
          out(i,2*d+2) = arma::sum(M_inv_disc % cum_momenta % m);
          
          //2) compute the virial criterion
          
          // save the last value of the virial
          delta_virial = virial;
          
          // compute the new virial
          virial = sum(theta % m);
          
          // approximate the exchange rate
          delta_virial = (virial - delta_virial) / eps;
          
          // update the log sum exp of the virial log virial exchange rates
          add_sign_log_sum_exp(log_abs_sum_virial,
                               delta_virial_sign,
                               std::log(std::abs(delta_virial)) - out(i,2*d+1),
                               segno(delta_virial));
          
          // cumulate the Hamiltonians
          sum_H = log_add_exp(sum_H,-out(i,2*d+1));
          
          // save the virial exhaustion criterion
          out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
          
        }
        
      }else if(k == 0){
        
        // initialize the cumulative momenta vector
        cum_momenta = m0;
        
        // set the current acceptance rate
        out(0,2*d+4) = 1.0;
        
        // compute the initial potential energy of the system
        out(0,2*d) = Rcpp::as<double>(nlp(theta0,args,true));
        
        // compute the hamiltonian for the function
        H = out(0,2*d) +
          0.5*arma::dot(arma::square(m0),M_inv_cont);
        out(0,2*d+1) = H;
        
        // add the NUTS criterion
        out(0,2*d+2) = 1.0;
        
        // add the virial exhaustion criterion
        //out(0,2*d+2) = arma::sum(m % theta);
        
        // initialize the log_abs_sum del virial and its sign
        //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
        log_abs_sum_virial = -arma::datum::inf;
        //delta_virial_sign = segno(out(0,2*d+2));
        delta_virial_sign = 0.0;
        
        // loop for each iteration
        for( unsigned int i = 1; i < max_it; i++){
          
          //compute the gradient
          arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
          
          //if the gradient is finite we can continue
          if(grad.is_finite()){
            
            //continuous momentum update by half step size
            m -= 0.5 * eps * grad;
            
            //continuous parameter update by one step size
            theta += eps * M_inv_cont % m;
            
            //compute the gradient
            grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
            
            //if the gradient is finite we can continue
            if(grad.is_finite()){
              
              //continuous momentum update by half step size
              m -= eps * grad;
              
              // compute the potential energy of the system
              out(i,2*d) = Rcpp::as<double>(nlp(theta,args,true));
              
              //compute the Hamiltonian of the system
              out(i,2*d+1) = out(i,2*d) + 
                0.5*arma::dot(arma::square(m),M_inv_cont); 
              
              //let's make sure it's not NaN, in which case let's set it equal to +Inf
              if(!std::isfinite(out(i,2*d))){
                out(i,2*d) = out(i,2*d+1) = arma::datum::inf;
              }
              
              //let's check if there is a divergent transition
              if( (out(i,2*d+1) - H) > 1000){
                
                //add the divergent transition to the global matrix
                theta -= 0.5 * eps * M_inv_cont % m;
                
                //report the divergent transition
                out(i,2*d+5) = 1.0;
                
              }else{
                //report the metropolis acceptance rate
                out(i,2*d+4) = std::exp(H-out(i,2*d+1));
              }
            }else{
              //add the divergent transition to the global matrix
              theta -= 0.5 * eps * M_inv_cont % m;
              
              //report the divergent transition
              out(i,2*d+5) = 1.0;    
            }
            
          }else{
            //report the divergent transition
            out(i,2*d+5) = 1.0;
          }
          
          // save the evolution of the flow in the phase space
          out.row(i).subvec(0,d-1) = theta.t();
          out.row(i).subvec(d,2*d-1) = m.t();
          
          //1) compute the NUTS criterion
          
          // cumulate the momenta vector
          cum_momenta += m;
          
          // compute the criterion
          out(i,2*d+2) = arma::sum(M_inv_cont % cum_momenta % m);
          
          //2) compute the virial criterion
          
          // save the last value of the virial
          delta_virial = virial;
          
          // compute the new virial
          virial = sum(theta % m);
          
          // approximate the exchange rate
          delta_virial = (virial - delta_virial) / eps;
          
          // update the log sum exp of the virial log virial exchange rates
          add_sign_log_sum_exp(log_abs_sum_virial,
                               delta_virial_sign,
                               std::log(std::abs(delta_virial)) - out(i,2*d+1),
                               segno(delta_virial));
          
          // cumulate the Hamiltonians
          sum_H = log_add_exp(sum_H,-out(i,2*d+1));
          
          // save the virial exhaustion criterion
          out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
        }
        
      }else{
        
        // initialize the cumulative momenta vector
        cum_momenta.subvec(0,d-k-1) = m0.subvec(0,d-k-1);
        cum_momenta.subvec(d-k,d-1) = arma::sign(m0.subvec(d-k,d-1));
        
        // set the current acceptance rate equal to one
        out.row(0).subvec(2*d+4,2*d+4+k) = arma::ones<arma::rowvec>(k+1);
        
        // compute the initial potential energy of the system
        out(0,2*d) = Rcpp::as<double>(nlp(theta0,args,true));
        
        // add the hamiltonian for the function
        H = out(0,2*d) + 
          0.5*arma::dot(arma::square(m0.subvec(0,d-k-1)),M_inv_cont) + 
          arma::dot(arma::abs(m0.subvec(d-k,d-1)),M_inv_disc);
        out(0,2*d+1) = H;
        
        // add the NUTS criterion
        out(0,2*d+2) = 1.0;
        
        // add the virial exhaustion criterion
        //out(0,2*d+2) = arma::sum(m % theta);
        
        // initialize the log_abs_sum del virial and its sign
        //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
        log_abs_sum_virial = -arma::datum::inf;
        //delta_virial_sign = segno(out(0,2*d+2));
        delta_virial_sign = 0.0;
        
        double U;
        double theta_old;
        double delta_U;
        unsigned int j;
        
        // loop for each iteration
        for( unsigned int i = 1; i < max_it; i++){
          
          //compute the gradient
          arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
          
          //if the gradient is finite we can continue
          if(grad.is_finite()){
            
            //continuous momentum update by half step size
            m.subvec(0,d-k-1) -= 0.5 * eps * grad;
            
            //continuous parameter update by half step size
            theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
            
            //compute the value of the new potential energy
            U = Rcpp::as<double>(nlp(theta,args,true));
            
            // if the potential energy is finite then we continue
            if(std::isfinite(U)){
              
              //permute the order of the discrete parameters
              idx_disc = arma::shuffle(idx_disc);
              
              //loop for every discontinuous component
              for(unsigned int ii = 0; ii < k; ii++){
                
                //set the current index
                j = idx_disc(ii);
                
                //modify the discrete parameter
                theta_old = theta(j);
                
                theta(j) = theta_old + eps * M_inv_disc(j-d+k) * segno(m(j));
                
                //calculation of the difference in potential energy
                delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
                
                //calculation of the Metropolis acceptance rate
                out(i,d+5+j+k) = std::exp(-delta_U);
                
                //refraction or reflection?
                if( M_inv_disc(j-d+k) * std::abs(m(j)) > delta_U ){
                  
                  //refraction
                  m(j) -= segno(m(j)) * delta_U / M_inv_disc(j-d+k);
                  U += delta_U;
                  
                }else{
                  
                  //reflection
                  theta(j) = theta_old;
                  m(j) *= -1.0;
                  
                  //report it
                  out(i,d+5+j+2*k) = 1.0;
                  
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
                
                // compute the potential energy of the current system
                out(i,2*d) = Rcpp::as<double>(nlp(theta,args,true));
                
                //compute the hamiltonian
                out(i,2*d+1) = out(i,2*d) + 
                  0.5*arma::dot(arma::square(m.subvec(0,d-k-1)),M_inv_cont) + 
                  arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
                
                //let's make sure it's not NaN, in which case let's set it equal to -Inf
                if(!std::isfinite(out(i,2*d))){
                  out(i,2*d) = out(i,2*d+1) = arma::datum::inf;
                }
                
                //let's check if there is a divergent transition
                if( (out(i,2*d+1) - H) > 1000){
                  
                  //add the divergent transition to the global matrix
                  theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
                  
                  //report the divergent transition
                  out(i,2*d+5+2*k) = 1.0;
                  
                }else{
                  //update the metropolis acceptance rate
                  out(i,2*d+4) += std::exp(H-out(i,2*d+1));
                }
              }else{
                
                //add the divergent transition to the global matrix
                theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
                
                //report the divergent transition
                out(i,2*d+5+2*k) = 1.0;
              }
            }else{
              
              //add the divergent transition to the global matrix
              theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont % m.subvec(0,d-k-1);
              
              //report the divergent transition
              out(i,2*d+5+2*k) = 1.0;
            }
          }else{
            //add the divergent transition to the global matrix
            
            //report the divergent transition
            out(i,2*d+5+2*k) = 1.0;
          }
          
          // save the evolution of the flow in the phase space
          out.row(i).subvec(0,d-1) = theta.t();
          out.row(i).subvec(d,2*d-1) = m.t();
          
          //1) compute the NUTS criterion
          
          // cumulate the momenta vector
          cum_momenta.subvec(0,d-k-1) += m.subvec(0,d-k-1);
          cum_momenta.subvec(d-k,d-1) += arma::sign(m.subvec(d-k,d-1));
          
          // compute the criterion
          out(i,2*d+2) = arma::sum(M_inv_cont % cum_momenta.subvec(0,d-k-1) % m.subvec(0,d-k-1));
          out(i,2*d+2) += arma::sum(M_inv_disc % cum_momenta.subvec(d-k,d-1) % arma::sign(m.subvec(d-k,d-1)));
          
          //2) compute the virial criterion
          
          // save the last value of the virial
          delta_virial = virial;
          
          // compute the new virial
          virial = sum(theta % m);
          
          // approximate the exchange rate
          delta_virial = (virial - delta_virial) / eps;
          
          // update the log sum exp of the virial log virial exchange rates
          add_sign_log_sum_exp(log_abs_sum_virial,
                               delta_virial_sign,
                               std::log(std::abs(delta_virial)) - out(i,2*d+1),
                               segno(delta_virial));
          
          // cumulate the Hamiltonians
          sum_H = log_add_exp(sum_H,-out(i,2*d+1));
          
          // save the virial exhaustion criterion
          out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
          
        }
        
      }
      
    }else{
      
      // dense matrix case
      
      // compute the inverses
      arma::mat M_inv_cont = arma::inv(Rcpp::as<arma::mat>(M_cont));
      arma::vec M_inv_disc = 1/arma::sqrt(Rcpp::as<arma::vec>(M_disc));
      
      if(k == 0){
        
        // initialize the cumulative momenta vector
        cum_momenta = m0;
        
        // set the current acceptance rate
        out(0,2*d+4) = 1.0;
        
        // compute the initial potential energy of the system
        out(0,2*d) = Rcpp::as<double>(nlp(theta0,args,true));
        
        // compute the hamiltonian for the function
        H = out(0,2*d) +
          0.5*arma::dot(m0,M_inv_cont * m0);
        out(0,2*d+1) = H;
        
        // add the NUTS criterion
        out(0,2*d+2) = 1.0;
        
        // add the virial exhaustion criterion
        //out(0,2*d+2) = arma::sum(m % theta);
        
        // initialize the log_abs_sum del virial and its sign
        //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
        log_abs_sum_virial = -arma::datum::inf;
        //delta_virial_sign = segno(out(0,2*d+2));
        delta_virial_sign = 0.0;
        
        // loop for each iteration
        for( unsigned int i = 1; i < max_it; i++){
          
          //compute the gradient
          arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
          
          //if the gradient is finite we can continue
          if(grad.is_finite()){
            
            //continuous momentum update by half step size
            m -= 0.5 * eps * grad;
            
            //continuous parameter update by half step size
            theta += eps * M_inv_cont * m;
            
            //compute the gradient
            grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
            
            //if the gradient is finite we can continue
            if(grad.is_finite()){
              
              //continuous momentum update by half step size
              m -= 0.5 * eps * grad;
              
              // compute the potential energy of the current system
              out(i,2*d) = Rcpp::as<double>(nlp(theta,args,true));
              
              //compute the Hamiltonian of the new system
              out(i,2*d+1) = out(i,2*d) + 
                0.5*arma::dot(m,M_inv_cont * m);
              
              //let's make sure it's not NaN, in which case let's set it equal to +Inf
              if(!std::isfinite(out(i,2*d))){
                out(i,2*d) = out(i,2*d+1) = arma::datum::inf;
              }
              
              //let's check if there is a divergent transition
              if( (out(i,2*d+1) - H) > 1000){
                
                //add the divergent transition to the global matrix
                theta -= 0.5 * eps * M_inv_cont * m;
                
                //report the divergent transition
                out(i,2*d+5) = 1.0;
                
              }else{
                //report the metropolis acceptance rate
                out(i,2*d+4) = std::exp(H-out(i,2*d+1));
              }
            }else{
              //add the divergent transition to the global matrix
              theta -= 0.5 * eps * M_inv_cont * m;
              
              //report the divergent transition
              out(i,2*d+5) = 1.0;    
            }
            
          }else{
            //report the divergent transition
            out(i,2*d+5) = 1.0;
          }
          
          // save the evolution of the flow in the phase space
          out.row(i).subvec(0,d-1) = theta.t();
          out.row(i).subvec(d,2*d-1) = m.t();
          
          //1) compute the NUTS criterion
          
          // cumulate the momenta vector
          cum_momenta += m;
          
          // compute the criterion
          out(i,2*d+2) = arma::sum(M_inv_cont * cum_momenta % m);
          
          //2) compute the virial criterion
          
          // save the last value of the virial
          delta_virial = virial;
          
          // compute the new virial
          virial = sum(theta % m);
          
          // approximate the exchange rate
          delta_virial = (virial - delta_virial) / eps;
          
          // update the log sum exp of the virial log virial exchange rates
          add_sign_log_sum_exp(log_abs_sum_virial,
                               delta_virial_sign,
                               std::log(std::abs(delta_virial)) - out(i,2*d+1),
                               segno(delta_virial));
          
          // cumulate the Hamiltonians
          sum_H = log_add_exp(sum_H,-out(i,2*d+1));
          
          // save the virial exhaustion criterion
          out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
          
        }
        
      }else if (k == d){
        
        Rcpp::stop("Dense continuous mass matrix make no sense if all the components are discontinuous!");
        
      }else{
        
        // initialize the cumulative momenta vector
        cum_momenta.subvec(0,d-k-1) = m.subvec(0,d-k-1);
        cum_momenta.subvec(d-k,d-1) = arma::sign(m.subvec(d-k,d-1));
        
        // set the current acceptance rate equal to one
        out.row(0).subvec(2*d+4,2*d+4+k) = arma::ones<arma::rowvec>(k+1);
        
        // compute the initial potential energy of the system
        out(0,2*d) = Rcpp::as<double>(nlp(theta0,args,true));
        
        // add the hamiltonian for the function
        H = out(0,2*d) + 
          0.5*arma::dot(m0.subvec(0 ,d-k-1),M_inv_cont * m0.subvec(0 ,d-k-1)) + 
          arma::dot(arma::abs(m0.subvec(d-k,d-1)),M_inv_disc);
        out(0,2*d+1) = H;
        
        // add the NUTS criterion
        out(0,2*d+2) = 1.0;
        
        // add the virial exhaustion criterion
        //out(0,2*d+2) = arma::sum(m % theta);
        
        // initialize the log_abs_sum del virial and its sign
        //log_abs_sum_virial = std::log(std::abs(out(0,2*d+2))) - H;
        log_abs_sum_virial = -arma::datum::inf;
        //delta_virial_sign = segno(out(0,2*d+2));
        delta_virial_sign = 0.0;
        
        double U;
        double theta_old;
        double delta_U;
        unsigned int j;
        
        // loop for each iteration
        for( unsigned int i = 1; i < max_it; i++){
          
          //compute the gradient
          arma::vec grad = Rcpp::as<arma::vec>(nlp(theta,args,false));
          
          //if the gradient is finite we can continue
          if(grad.is_finite()){
            
            //continuous momentum update by half step size
            m.subvec(0,d-k-1) -= 0.5 * eps * grad;
            
            //continuous parameter update by half step size
            theta.subvec(0,d-k-1) += 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
            
            //compute the value of the new potential energy
            U = Rcpp::as<double>(nlp(theta,args,true));
            
            // if the potential energy is finite then we continue
            if(std::isfinite(U)){
              
              //permute the order of the discrete parameters
              idx_disc = arma::shuffle(idx_disc);
              
              //loop for every discontinuous component
              for(unsigned int ii = 0; ii < k; ii++){
                
                //set the current index
                j = idx_disc(ii);
                
                //modify the discrete parameter
                theta_old = theta(j);
                
                theta(j) = theta_old + eps * M_inv_disc(j-d+k) * segno(m(j));
                
                //calculation of the difference in potential energy
                delta_U = Rcpp::as<double>(nlp(theta,args,true)) - U;
                
                //calculation of the Metropolis acceptance rate
                out(i,d+5+j+k) = std::exp(-delta_U);
                
                //refraction or reflection?
                if( M_inv_disc(j-d+k) * std::abs(m(j)) > delta_U ){
                  
                  //refraction
                  m(j) -= segno(m(j)) * delta_U / M_inv_disc(j-d+k);
                  U += delta_U;
                  
                }else{
                  
                  //reflection
                  theta(j) = theta_old;
                  m(j) *= -1.0;
                  
                  //report it
                  out(i,d+5+j+2*k) = 1.0;
                  
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
                
                // compute the potential energy of the new system
                out(i,2*d) = Rcpp::as<double>(nlp(theta,args,true));
                
                //compute the hamiltonian
                out(i,2*d+1) = out(i,2*d) + 
                  0.5*arma::dot(m.subvec(0 ,d-k-1),M_inv_cont * m.subvec(0 ,d-k-1)) + 
                  arma::dot(arma::abs(m.subvec(d-k,d-1)),M_inv_disc);
                
                //let's make sure it's not NaN, in which case let's set it equal to +Inf
                if(!std::isfinite(out(i,2*d))){
                  out(i,2*d) = out(i,2*d + 1) = arma::datum::inf;
                }
                
                //let's check if there is a divergent transition
                if( (out(i,2*d+1) - H) > 1000){
                  
                  //add the divergent transition to the global matrix
                  theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
                  
                  //report the divergent transition
                  out(i,2*d+5+2*k) = 1.0;
                  
                }else{
                  //update the metropolis acceptance rate
                  out(i,2*d+4) += std::exp(H-out(i,2*d+1));
                }
              }else{
                
                //add the divergent transition to the global matrix
                theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
                
                //report the divergent transition
                out(i,2*d+5+2*k) = 1.0;
              }
            }else{
              
              //add the divergent transition to the global matrix
              theta.subvec(0,d-k-1) -= 0.5 * eps * M_inv_cont * m.subvec(0,d-k-1);
              
              //report the divergent transition
              out(i,2*d+5+2*k) = 1.0;
            }
          }else{
            
            //report the divergent transition
            out(i,2*d+5+2*k) = 1.0;
          }
          
          // save the evolution of the flow in the phase space
          out.row(i).subvec(0,d-1) = theta.t();
          out.row(i).subvec(d,2*d-1) = m.t();
          
          //1) compute the NUTS criterion
          
          // cumulate the momenta vector
          cum_momenta.subvec(0,d-k-1) += m.subvec(0,d-k-1);
          cum_momenta.subvec(d-k,d-1) += arma::sign(m.subvec(d-k,d-1));
          
          // compute the criterion
          out(i,2*d+2) = arma::sum(M_inv_cont * cum_momenta.subvec(0,d-k-1) % m.subvec(0,d-k-1));
          out(i,2*d+2) += arma::sum(M_inv_disc % cum_momenta.subvec(d-k,d-1) % arma::sign(m.subvec(d-k,d-1)));
          
          //2) compute the virial criterion
          
          // save the last value of the virial
          delta_virial = virial;
          
          // compute the new virial
          virial = sum(theta % m);
          
          // approximate the exchange rate
          delta_virial = (virial - delta_virial) / eps;
          
          // update the log sum exp of the virial log virial exchange rates
          add_sign_log_sum_exp(log_abs_sum_virial,
                               delta_virial_sign,
                               std::log(std::abs(delta_virial)) - out(i,2*d+1),
                               segno(delta_virial));
          
          // cumulate the Hamiltonians
          sum_H = log_add_exp(sum_H,-out(i,2*d+1));
          
          // save the virial exhaustion criterion
          out(i,2*d+3) = std::exp(log_abs_sum_virial - sum_H - std::log(i+1));
          
        }
        
      }
      
    }
    
  }
  
  // set the first value of the virial equal to NA
  out(0,2*d+3) = NA_REAL;
  
  // coerce the output matrix as a list
  Rcpp::List df;
  Rcpp::CharacterVector df_names(out_dim);
  
  // assign the names to the columns
  for(unsigned int i = 0; i < d; i++){
    df_names[i] = position_names[i];
    df_names[d+i] = momentum_names[i];
  }
  df_names[2*d] = "Potential";
  df_names[2*d + 1] = "Hamiltonian";
  df_names[2*d + 2] = "NUTS";
  df_names[2*d + 3] = "Virial";
  df_names[out_dim - 1] = "Divergent";
  
  if( k == d){
    
    // add the alphas and the reflection dummies
    for(unsigned int i = 0; i < d; i++){
      df_names[2*d+4+i] = "alpha_" + position_names[i];
      df_names[3*d+4+i] = "reflection_" + position_names[i];
    }
    
  }else if(k == 0){
    
    // add the global acceptance probability
    df_names[2*d + 4] = "alpha";
    
  }else{
    
    // add the global acceptance probability
    df_names[2*d + 4] = "alpha";
    
    // add the alphas and the reflection dummies
    for(unsigned int i = 0; i < k; i++){
      df_names[2*d+5+i] = "alpha_" + position_names[i];
      df_names[2*d+5+k+i] = "reflection_" + position_names[i];
    }
    
  }
  
  // insert all the desired quantities
  for(unsigned int i = 0; i < out.n_cols; i++){
    df[Rcpp::as<string>(df_names[i])] = out.col(i);
  }
  
  // coerce the list to data frame
  df.attr("names") = df_names;
  df.attr("class") = "data.frame";
  df.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -out.n_rows);
  
  Rcpp::DataFrame df_out = df;
  
  df_out.attr("class") = Rcpp::CharacterVector::create("XDNUTS.trajectories","data.frame");
  
  // return the output dataframe
  return df_out;
  
}
