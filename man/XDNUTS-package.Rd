\name{XDNUTS-package}
\alias{XDNUTS-package}
\alias{XDNUTS}
\docType{package}
\title{
\packageTitle{XDNUTS}
}
\description{
\packageDescription{XDNUTS}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{XDNUTS}
\packageIndices{XDNUTS}
The package allows to use a more efficient version of the Discontinuous Hamiltonian Monte Carlo proposed in \insertCite{nishimura2020discontinuous}{XDNUTS}, thanks to the use of recycled samples from each trajectory \insertCite{Nishimura_2020}{XDNUTS} and a termination criterion for identyfing the optimal discrete integration time of each trajectory \insertCite{betancourt2016identifying}{XDNUTS}. No models are at disposal, so the user must specify one through the definition of the function \code{nlp}. This function must evaluate the negative log posterior of the model and its gradient with respect to the first \eqn{d-k} parameters. \eqn{d} is the model dimension, while \eqn{k} is the number of parameters for which the sampling scheme will be based on the method described in \insertCite{nishimura2020discontinuous}{XDNUTS}. This method was born for treating discontinuous components but it is applicable to continuous one too. \code{nlp} must be a function with 3 arguments:
\itemize{
\item{the vector of parameters, current state of the chain/trajectory, for which the negative log posterior or its gradient must be evaluated.}
\item{a list object that contains the necessary argouments, namely data and hyperparameters.}
\item{a boolean value, \code{TRUE} to evaluate only the negative log posterior of the models, 
\code{FALSE} to evaluate its gradient with respect to the continuous components of the posterior.}
}
The available algorithms are the following
\describe{
\item{NUTS}{No U-Turn Sampler of \insertCite{hoffman2014no}{XDNUTS}.}
\item{XHMC}{Hamiltonian Monte Carlo with a termination criterion based on the exhustion of the virial \insertCite{betancourt2016identifying}{XDNUTS} which require the specification of a threshold.}
\item{HMC}{Hamiltonian Monte Carlo with trajectory length varying uniformly inside a user specified interval. Instead of proposing the last value of each trajectories a sample is drawn uniformly from them. Reference can be found in \insertCite{betancourt2017conceptual}{XDNUTS}.}
}
All of them are embedded into the framework described in \insertCite{nishimura2020discontinuous}{XDNUTS} which allows the use of Hamiltonian Monte Carlo with discontinuous posterior and hence to discrete parameter space by the definition of a step function shape density.
}
\author{
\packageAuthor{XDNUTS}

Maintainer: \packageMaintainer{XDNUTS}
}
\references{
\insertRef{hoffman2014no}{XDNUTS}\cr\cr
\insertRef{betancourt2016identifying}{XDNUTS}\cr\cr
\insertRef{betancourt2017conceptual}{XDNUTS}\cr\cr
\insertRef{nishimura2020discontinuous}{XDNUTS}\cr\cr
\insertRef{Nishimura_2020}{XDNUTS}
}


