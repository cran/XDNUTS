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
    if(tmp.length() != 2){
      Rcpp::stop("'delta' must be a vector of length two!");
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
  if(delta_val(0) <= 0 || delta_val(0) >= 1 || delta_val(1) <= 0 || delta_val(1) >= 1 || delta_val.size() != 2){
    Rcpp::stop("'delta' must contain two scalar in the interval (0,1)!");
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
