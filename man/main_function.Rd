% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{main_function}
\alias{main_function}
\title{Function to generate a Markov chain for both continuous and discontinuous posterior distributions.}
\usage{
main_function(
  theta0,
  nlp,
  args,
  k,
  N,
  K,
  tau,
  L,
  thin,
  chain_id,
  verbose,
  control
)
}
\arguments{
\item{theta0}{a vector of length-\eqn{d} containing the starting point of the chain.}

\item{nlp}{a function object that takes three arguments:
\describe{\item{par}{a vector of length-\eqn{d} containing the value of the parameters.}
\item{args}{a list object that contains the necessary arguments, namely data and hyperparameters.}
\item{eval_nlp}{a boolean value, \code{TRUE} for evaluating only the model\'s negative log posterior, 
\code{FALSE} to evaluate the gradient with respect to the continuous components of the posterior.}}}

\item{args}{the necessary arguments to evaluate the negative log posterior and its gradient.}

\item{k}{the number of parameters that induce a discontinuity in the posterior distribution.}

\item{N}{integer containing the number of post warm-up samples to evaluate.}

\item{K}{integer containing the number of recycled samples from each trajectory during the warm-up phase or beyond.}

\item{tau}{the threshold for the exhaustion termination criterion described in \insertCite{betancourt2016identifying}{XDNUTS}.}

\item{L}{the desired length of the trajectory of classic Hamiltonian Monte Carlo algorithm.}

\item{thin}{integer containing the number of samples to discard in order to produce a final iteration of the chain.}

\item{chain_id}{the identification number of the chain.}

\item{verbose}{a boolean value that controls whether to print all the information regarding the sampling process.}

\item{control}{an object of class \code{control_xdnuts} containing the specifications for the algorithm.
See the \link{set_parameters} function for detail.}
}
\value{
a named list containing: \describe{
\item{values}{a \eqn{N \times d} matrix containing the sample from the
target distribution (if convergence has been reached).}
\item{energy}{a vector of length-\eqn{N} containing the Markov Chain of the energy level sets.}
\item{delta_energy}{a vector of length-\eqn{N} containing the Markov Chain of the first difference energy level sets.}
\item{step_size}{a vector of length-\eqn{N} containing the sampled step size used for each iteration.}
\item{step_length}{a vector of length-\eqn{N} containing the length of each trajectory of the chain.}
\item{alpha}{a vector of length-\eqn{k + 1} containing the estimate of the Metropolis acceptance probabilities.
The first element of the vector is the estimated global acceptance probability. The remaining k elements are the 
estimate rate of reflection for each parameters which travels coordinate-wise through some discontinuity.}
\item{warm_up}{a \eqn{N_{adapt} \times d} matrix containing the sample of the chain
coming from the warm-up phase. If \code{keep_warm_up = FALSE} inside the \code{control}
argument, nothing is returned.}
\item{div_trans}{a \eqn{M \times d} matrix containing the locations where a divergence has been 
encountered during the integration of Hamilton equation. Hopefully \eqn{M \ll N}, and even better if \eqn{M = 0}.}
\item{M_cont}{the Mass Matrix of the continuous components estimated during the warm-up phase.
              Based on the \code{M_type} value of the \code{control} arguments, this could be an empty object, a vector or a matrix.}
\item{M_disc}{the Mass Matrix of the discontinuous components estimated during the warm-up phase.
              Based on the \code{M_type} value of the \code{control} arguments, this could be an empty object or a vector.}}
}
\description{
The function allows to generate a single Markov Chain for sampling from both continuous and discontinuous
 posterior distributions using a plethora of algorithms. Classic Hamiltonian Monte Carlo \insertCite{duane1987hybrid}{XDNUTS} ,
  NUTS \insertCite{hoffman2014no}{XDNUTS} and XHMC \insertCite{betancourt2016identifying}{XDNUTS} are embedded into the framework
described in \insertCite{nishimura2020discontinuous}{XDNUTS}, which allows to deal with such posteriors.
Furthermore, for each method, it is possible to recycle samples from the trajectories using
the method proposed by \insertCite{Nishimura_2020}{XDNUTS}.
This is used to improve the estimation of the mass matrix during the warm-up phase
without requiring significant additional computational costs.
This function should not be used directly, but only through the user interface provided by \link{xdnuts}.
}
\references{
\insertAllCited{}
}
