# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Function that regulates the specifications of the \link{xdnuts} function.
#' @param N_init1 an integer that regulates the number of samples used to
#' adapt the step size.
#' @param N_adapt an integer that regulates the number of samples used to
#' estimate the Mass Matrix with fixed step size.
#' @param N_init2 an integer that regulates the number of samples used to
#'  adapt the step size after the estimation of the Mass Matrix.
#' @param burn_adapt_ratio a numeric scalar \eqn{\in (0,1]} indicating the ratio of warm-up
#' samples to discard in order to estimate the covariance matrix of the parameters.
#' @param keep_warm_up a logical scalar that determines whether the warm-up samples should be returned.
#' @param recycle_only_init a logical value which disables the recycling of the
#' samples from each trajectory once the warm-up phase has terminated.
#' @param max_treedepth an integer that regulates the maximum depth of
#' the binary tree used to approximate Hamilton equation for the exploration
#' of each energy level set of the phase space.
#' @param max_treedepth_init an integer that controls the maximum depth of
#' the binary tree during the step-size adaptation phase. Setting a smaller value 
#' can help avoid wasting valuable time on low-probability areas with suboptimal 
#' algorithm parameters.
#' @param eps_jitter a numeric scalar which regulates the amount of jittering
#' used to perturb the value of the step size for each iteration of the chain
#' after the warm-up phase.
#' @param L_jitter an integer scalar that regulates the amount of jittering used to perturb the
#'  value of the trajectory length if this is specified to be constant.
#'  This occurs when the classic Hamiltonian Monte Carlo algorithm is used through the 
#'  \code{method = "HMC"} option in the \link{xdnuts} function. If \code{L_jitter} \eqn{\geq 1}
#'  each trajectory length is sampled uniformly inside the interval [L - L_jitter , L + L_jitter].
#' @param gamma a numeric value that, in the Nesterov Dual Averaging algorithm, regulates 
#'  the sensitivity of the step size updating scheme to fluctuations in the estimate of the 
#'  mean Metropolis acceptance probability.
#' @param kappa a numeric value that regulates the vanishing of Nesterov Dual Averaging
#' algorithm for the estimation of the step size.
#' @param delta a vector containing the Metropolis acceptance probabilities, 
#'  including both the global and those related to potential differences. Default values are (0.8,0.6).
#'  If the second element of the vector is set to \code{NA}, then the step size calibration is conducted 
#'  solely through the global acceptance probabilities.
#' @param t0 an integer value that makes Nesterov Dual Averaging
#' algorithm for the estimation of the step size less sensitive to early iterations.
#' @param M_type a character value specifying the type of Mass Matrix to estimate:\itemize{
#' \item{\code{"identity"} no Mass Matrix estimation is done.}
#' \item{\code{"diagonal"} a diagonal Mass Matrix is estimated during the warm-up phase.}
#' \item{\code{"dense"} a full dense Mass Matrix is estimated during the warm-up phase.}
#' }
#' @param refresh a numeric scalar bounded in \eqn{(0,1)} which regulates the update frequency of
#' the displayed sampling process state. Default values is 0.1, meaning every 10\% of the total samples.
#' @param l_eps_init a numeric scalar containing the logarithm of the initial value for the step size
#' used to approximate Hamilton differential equation for phase space exploration.
#' @param different_stepsize a boolean value indicating where the adaptation scheme should adapt different step size. 
#'  If \code{TRUE}, a global step size is adapted via Nesterov Dual Averaging algorithm. 
#'  At the same time, for each empirical reflection rate of each component treated as discontinuous the same
#'  algorithm is exploited and the difference between these is obtained through the updating of the discontinuous
#'  components Mass Matrix. Default value is \code{FALSE}.
#' @param mu a numeric scalar containing the value to which the step size is shrunken during the warm-up phase.
#' @param M_cont a vector of length-\eqn{d-k} if \code{M_type = "diagonal"} or a \eqn{(d-k) \times (d-k)} matrix
#' if \code{M_type = "dense"} containing an initial estimate for the Mass Matrix
#' (the inverse of the parameters covariance matrix).
#' If you want to keep it fixed, they should specify \code{N_adapt = 0}.
#' @param M_disc a vector of length-\eqn{k} if \code{M_type = "diagonal"} or 
#' \code{M_type = "dense"} containing an initial estimate for the Mass Matrix
#' (the inverse of the parameters covariances).
#' If one wants to keep it fixed, they should specify \code{N_adapt = 0}.
#' @return an object of class \code{control_xdnuts} containing a named list with all the above parameters.
#'
#' @export set_parameters
set_parameters <- function(N_init1 = 50L, N_adapt = 200L, N_init2 = 75L, burn_adapt_ratio = 0.1, keep_warm_up = FALSE, recycle_only_init = TRUE, max_treedepth = 10L, max_treedepth_init = 10L, eps_jitter = 0.1, L_jitter = 3L, gamma = 0.05, kappa = 0.75, delta = NULL, t0 = 10L, M_type = "dense", refresh = 0.1, l_eps_init = NA_real_, different_stepsize = FALSE, mu = NA_real_, M_cont = NULL, M_disc = NULL) {
    .Call(`_XDNUTS_set_parameters`, N_init1, N_adapt, N_init2, burn_adapt_ratio, keep_warm_up, recycle_only_init, max_treedepth, max_treedepth_init, eps_jitter, L_jitter, gamma, kappa, delta, t0, M_type, refresh, l_eps_init, different_stepsize, mu, M_cont, M_disc)
}

#' Function to generate a Markov chain for both continuous and discontinuous posterior distributions.
#' @description The function allows to generate a single Markov Chain for sampling from both continuous and discontinuous
#'  posterior distributions using a plethora of algorithms. Classic Hamiltonian Monte Carlo \insertCite{duane1987hybrid}{XDNUTS} ,
#'   NUTS \insertCite{hoffman2014no}{XDNUTS} and XHMC \insertCite{betancourt2016identifying}{XDNUTS} are embedded into the framework
#' described in \insertCite{nishimura2020discontinuous}{XDNUTS}, which allows to deal with such posteriors.
#' Furthermore, for each method, it is possible to recycle samples from the trajectories using
#' the method proposed by \insertCite{Nishimura_2020}{XDNUTS}.
#' This is used to improve the estimation of the mass matrix during the warm-up phase
#' without requiring significant additional computational costs.
#' This function should not be used directly, but only through the user interface provided by \link{xdnuts}.
#' @param theta0 a vector of length-\eqn{d} containing the starting point of the chain.
#' @param nlp a function object that takes three arguments:
#' \describe{\item{par}{a vector of length-\eqn{d} containing the value of the parameters.}
#' \item{args}{a list object that contains the necessary arguments, namely data and hyperparameters.}
#' \item{eval_nlp}{a boolean value, \code{TRUE} for evaluating only the model\'s negative log posterior, 
#' \code{FALSE} to evaluate the gradient with respect to the continuous components of the posterior.}}
#' @param args the necessary arguments to evaluate the negative log posterior and its gradient.
#' @param k the number of parameters that induce a discontinuity in the posterior distribution.
#' @param N integer containing the number of post warm-up samples to evaluate.
#' @param K integer containing the number of recycled samples from each trajectory during the warm-up phase or beyond.
#' @param tau the threshold for the exhaustion termination criterion described in \insertCite{betancourt2016identifying}{XDNUTS}.
#' @param L the desired length of the trajectory of classic Hamiltonian Monte Carlo algorithm.
#' @param thin integer containing the number of samples to discard in order to produce a final iteration of the chain.
#' @param chain_id the identification number of the chain.
#' @param verbose a boolean value that controls whether to print all the information regarding the sampling process.
#' @param control an object of class \code{control_xdnuts} containing the specifications for the algorithm.
#' See the \link{set_parameters} function for detail.
#' 
#' @return a named list containing: \describe{
#'\item{values}{a \eqn{N \times d} matrix containing the sample from the
#'target distribution (if convergence has been reached).}
#'\item{energy}{a vector of length-\eqn{N} containing the Markov Chain of the energy level sets.}
#'\item{delta_energy}{a vector of length-\eqn{N} containing the Markov Chain of the first difference energy level sets.}
#'\item{step_size}{a vector of length-\eqn{N} containing the sampled step size used for each iteration.}
#'\item{step_length}{a vector of length-\eqn{N} containing the length of each trajectory of the chain.}
#'\item{alpha}{a vector of length-\eqn{k + 1} containing the estimate of the Metropolis acceptance probabilities.
#' The first element of the vector is the estimated global acceptance probability. The remaining k elements are the 
#' estimate rate of reflection for each parameters which travels coordinate-wise through some discontinuity.}
#' \item{warm_up}{a \eqn{N_{adapt} \times d} matrix containing the sample of the chain
#' coming from the warm-up phase. If \code{keep_warm_up = FALSE} inside the \code{control}
#' argument, nothing is returned.}
#' \item{div_trans}{a \eqn{M \times d} matrix containing the locations where a divergence has been 
#' encountered during the integration of Hamilton equation. Hopefully \eqn{M \ll N}, and even better if \eqn{M = 0}.}
#' \item{M_cont}{the Mass Matrix of the continuous components estimated during the warm-up phase.
#'               Based on the \code{M_type} value of the \code{control} arguments, this could be an empty object, a vector or a matrix.}
#' \item{M_disc}{the Mass Matrix of the discontinuous components estimated during the warm-up phase.
#'               Based on the \code{M_type} value of the \code{control} arguments, this could be an empty object or a vector.}}
#'
#' @references
#'  \insertAllCited{}
#' 
#' @export main_function
main_function <- function(theta0, nlp, args, k, N, K, tau, L, thin, chain_id, verbose, control) {
    .Call(`_XDNUTS_main_function`, theta0, nlp, args, k, N, K, tau, L, thin, chain_id, verbose, control)
}

#' Function that approximate the Hamiltonian Flow for given starting values
#' of the position and momentum of a particle in the phase space defined by the 
#' kinetic and potential energy provided in input. 
#' @param theta0 a numeric vector of length \eqn{d} representing
#'  the starting position vector for the particle.
#' @param m0 a numeric vector of length \eqn{d} representing 
#' the starting momenta vector for the particle.
#' @param nlp a function object that evaluate the negative of the logarithm of 
#' a probability density function, and its gradient, i.e. the potential energy function of the system.
#' @param args a list object containing the arguments to be passed to the function \code{nlp}.
#' @param eps a numeric scalar indicating the step size for the \emph{leapfrog} integrator.
#' @param k an integer scalar indicating the number of discontinuous components of \code{theta0}.
#' @param M_cont either a vector or a squared matrix, of the same length/dimension of
#' the position/momenta vector, representing the continuous components mass matrix.
#' @param M_disc a vector of the same length of the position/momenta vector,
#'  representing the discontinuous components mass matrix.
#' @param max_it an integer value indicating the length of the trajectory. 
#' This quantity times \code{eps} is equal to the approximated integration time
#' of the Hamiltonian flow.
#' @return a data frame that summarizes the approximated Hamiltonian flow.
#' \itemize{
#' \item{The first \eqn{d} columns contain the particle position evolution.}
#' \item{The second \eqn{d} columns contain the particle momenta evolution.}
#' \item{The \eqn{2d + 1} column contains the Hamiltonian evolution.}
#' \item{The \eqn{2d + 2} column contains the evolution of the No U-Turn Sampler termination criterion.}
#' \item{The \eqn{2d + 3} column contains the evolution of the virial exhaustion termination criterion.}
#' \item{Acceptance and or refraction probabilities. This depends on the value of \code{k}.}
#' \item{Reflession dummy indicators.}
#' \item{Divergent transition dummy indicators.}
#' } 
#' @export trajectories
trajectories <- function(theta0, m0, nlp, args, eps, k, M_cont, M_disc, max_it) {
    .Call(`_XDNUTS_trajectories`, theta0, m0, nlp, args, eps, k, M_cont, M_disc, max_it)
}

