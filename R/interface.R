### FUNCTION FOR DISCONTINUOUS HAMILTONIAN MONTE CARLO
#' Discontinuous Hamiltonian Monte Carlo using both manual and automatic termination criteria.
#'
#' @description The function allows generating multiple Markov Chains for sampling from both continuous and discontinuous
#' posterior distributions using a variety of algorithms. Classic Hamiltonian Monte Carlo \insertCite{duane1987hybrid}{XDNUTS}, 
#' NUTS \insertCite{hoffman2014no}{XDNUTS}, and XHMC \insertCite{betancourt2016identifying}{XDNUTS} are embedded into the framework
#' described in \insertCite{nishimura2020discontinuous}{XDNUTS}, which allows dealing with such posteriors.
#' Furthermore, for each method, it is possible to recycle samples from the trajectories using
#' the method proposed by \insertCite{Nishimura_2020}{XDNUTS}.
#' This is used to improve the estimate of the Mass Matrix during the warm-up phase
#' without requiring a relevant additional computational cost.
#' 
#' @param theta0 a list containing the starting values for each chain. These starting values are vectors of length-\eqn{d}. 
#' The last \eqn{k \in [0,d]} elements refer to parameters which determine a discontinuity in the posterior distribution. 
#' @param nlp a function which evaluates the negative log posterior and its gradient with respect to 
#' parameters that do not induce any discontinuity in the posterior distribution (more generally, the first \eqn{d-k} parameters).
#' This function must take 3 arguments:
#' \describe{
#' \item{par}{a vector of length-\eqn{d} containing the parameter values.}
#' \item{args}{a list object that contains the necessary arguments, namely data and hyperparameters.}
#' \item{eval_nlp}{a boolean value, \code{TRUE} to evaluate only the negative log posterior of the models, 
#' \code{FALSE} to evaluate its gradient with respect to the continuous components of the posterior.}
#' }
#' 
#' @param args a list containing the inputs for the negative posterior function.
#' @param k an integer value that states the number of parameters that determines a discontinuity in the posterior distribution.
#' Actually, since the algorithm proposed in \insertCite{nishimura2020discontinuous}{XDNUTS} also works for the full continuous case,
#' \code{k} is the number of parameters specified by the user for which this algorithm is used.
#' @param tau the threshold for the virial termination criterion \insertCite{betancourt2016identifying}{XDNUTS}.
#' @param L the desired length of the trajectory of classic Hamiltonian Monte Carlo algorithm.
#' @param N the number of draws from the posterior distribution, after warm-up, for each chain. Default value is \code{1000}.
#' @param K the number of recycled samples per iteration used by default during the warm-up phase.
#' Default value is \code{3}. To recycle in the sampling phase too, specify \code{recycle_only_init = FALSE}
#' in the \code{control} argument above.
#' @param method a character value which defines the type of algorithm to exploit:\describe{
#' \item{\code{"NUTS"}}{applies the No U-Turn Sampler of \insertCite{hoffman2014no}{XDNUTS}.}
#' \item{\code{"XHMC"}}{applies the Exhaustion Hamiltonian Monte Carlo of \insertCite{betancourt2016identifying}{XDNUTS}.}
#' \item{\code{"HMC"}}{applies one of the classic version of Hamiltonian Monte Carlo algorithm,
#' in particular the one described in \insertCite{betancourt2017conceptual}{XDNUTS}, which samples from the trajectory instead of always returning the last value.}
#' }
#' @param thin the number of necessary and discarded samples to obtain a final iteration of one chain.
#' @param control an object of class \code{control_xdnuts}, output of the function \link{set_parameters}.
#' @param parallel a boolean value specifying whether the chains must be run in parallel. Default value is \code{FALSE}.
#' @param loadLibraries A character vector indicating the names of the packages to load on each cluster if
#' \code{parallel} is set to \code{TRUE}. 
#' @param loadRObject A character vector indicating the names of the R objects to load on each cluster if
#' \code{parallel} is set to \code{TRUE}.
#' @param verbose a boolean value for printing all the information regarding the sampling process.
#' @param hide a boolean value that omits the printing to the console if set to \code{TRUE}.
#'
#' @return a list of class \code{XDNUTS} containing \describe{
#' \item{chains}{a list of the same length of \code{theta0}, each element containing the output from the function \link{main_function}.}
#' \item{d}{the dimension of the parameter space.}
#' \item{k}{the number of parameters that lead to a discontinuous posterior distribution. 
#' Or, more generally, for which the algorithm of \insertCite{nishimura2020discontinuous}{XDNUTS} is exploited.}
#' \item{K}{the number of recycled samples for each iteration during the sampling phase.}
#' \item{N}{the number of posterior draws for each chain.}
#' \item{method}{the MCMC method used. This could be either "NUTS", "XHMC", or "HMC".}
#' \item{tau}{the threshold for the virial termination criterion \insertCite{betancourt2016identifying}{XDNUTS}. 
#' Only if \code{method = "XHMC"} this value is different from zero.}
#' \item{L}{the desired length of the trajectory of classic Hamiltonian Monte Carlo algorithm specified by the user.
#' This argument is necessary if \code{method = "HMC"}.}
#' \item{thin}{the number of discarded samples for every final iteration, specified by the user.}
#' \item{control}{an object of class \code{control_xdnuts}, output of the function \link{set_parameters} with arguments specified by the user.}
#' \item{verbose}{the boolean value specified by the user regarding the printing of the sampling process information.}
#' \item{parallel}{the boolean value specified by the user regarding parallel processing.}
#' }
#' 
#' 
#' @references 
#'  \insertAllCited{}
#'
#' @export xdnuts
xdnuts <- function(theta0,
                  nlp,
                  args,
                  k,
                  N = 1e3,
                  K = 3,
                  method = "NUTS",
                  tau = NULL,
                  L = NULL,
                  thin = 1,
                  control = set_parameters(),
                  parallel = FALSE,
                  loadLibraries = NULL,
                  loadRObject = NULL,
                  verbose = FALSE,
                  hide = FALSE){
  #require(purrr)
  
  #let's make sure that the first input is a list
  if(!is.list(theta0)){
    base::stop("'theta0' must be a list containing the initial value for each chain!")
  }
  
  #let's make sure that the initial value of the chains are admissible
  for(i in seq_along(theta0)){
    if(!is.finite(nlp(theta0[[i]],args,TRUE)) || any(!is.finite(nlp(theta0[[i]],args,FALSE)))){
      base::stop("Not an admissible starting value for chain ", i)
    }
  }
  
  #let's make sure that nlp is a function
  if(!is.function(nlp)){
    base::stop("'nlp' must be a function object!")
  }
  
  #let's make sure that args is a list
  if(!is.list(args)){
    base::stop("'args' must be a list object!")
  }
  
  #let's make sure that k doesn't exceed both 0 and d
  if(all(k < 0 | k > length(theta0[[1]])) || length(k) > 1){
    base::stop("'k' must be a scalar, bounded  between 0 and the total number of parameters!")
  }
  
  #let's make sure that the gradient is of the right length
  if(base::length(nlp(theta0[[1]],args,FALSE)) != base::length(theta0[[1]]) - k &&
     base::length(theta0[[1]]) != k){
    base::stop("The gradient of the function nlp must have d - k elements!")
  }
  
  #let's make sure that the sample size is adequate
  if(all(N <= 0) || length(N) > 1){
    base::stop("'N' must be an integer scalar greater than zero!")
  }
  
  #let's make sure that the number of recycled samples is adequate
  if(all(K <= 0) || length(K) > 1){
    base::stop("'K' must be an integer scalar greater than zero!")
  }
  
  #let's make sure that the method specified is included among those available
  if(! (method %in% c("NUTS","XHMC","HMC"))){
    base::stop("'method' must be either 'NUTS', 'XHMC' or 'HMC'!")
  }
  
  #let's make sure delta is right
  if( k == length(theta0[[1]]) && is.na(control$delta[2])){
    base::stop("Second element of 'delta' in 'control' can't be NA if thera are only discontinuous components!")
  }
  if( k != length(theta0[[1]]) && is.na(control$delta[1])){
    base::stop("First element of 'delta' in 'control' can't be NA if there are also continuous components!")
  }
  
  #once the method is specified set to zero the unnecessary arguments
  #and make sure that the necessary one are appropriate
  if(method == "NUTS"){
    tau <- 0
    L <- 0
  }else if(method == "XHMC"){
    L <- 0
    if(is.null(tau)){
      base::stop("'tau' must be specified!")
    }
    if(all(tau <= 0) || length(tau) > 1){
      base::stop("'tau' must be a scalar greater than zero!")
    }
  }else if(method == "HMC"){
    tau <- 0
    if(is.null(L)){
      base::stop("'L' must be specified!")
    }
    if(all(L <= 0) || length(L) > 1){
      base::stop("'L' must be a scalar integer greater than zero!")
    }
  }
  
  #let's make sure that the number of sample to discard is appropriate
  if(all(thin <= 0) || length(thin) > 1){
    base::stop("'thin' must be a scalar integer greater than zero!")
  }
  
  #let's make sure that the control arguments is of the type control_xdnuts
  #and not a simple list
  if(!base::inherits(control,"control_xdnuts")){
    base::stop("'control' must be an object of class control_xdnuts!")
  }
  
  #let's make sure that the parallel argument is logical
  if(!is.logical(parallel) || length(parallel) > 1){
    base::stop("'parallel' must be a logical scalar!")
  }
  
  #let's make sure that the names of the object to pass to each core
  #are either of character type or NULL type
  if(!is.character(loadLibraries) && !is.null(loadLibraries)){
    base::stop("'loadLibraries' must be a character vector!")
  }
  if(!is.character(loadRObject) && !is.null(loadRObject)){
    base::stop("'loadRObject' must be a character vector!")
  }
  
  #let's make sure that the verbose argument is logical
  if(!is.logical(verbose) || length(verbose) > 1){
    base::stop("'verbose' must be a logical scalar!")
  }
  
  #let's make sure that the hide argument is logical
  if(!is.logical(hide) || length(hide) > 1){
    base::stop("'hide' must be a logical scalar!")
  }
  
  #get the number of chains from the length of the list
  n_chains <- length(theta0)
  
  #get the name of the coordinate from the first element of the list
  nomi <- names(theta0[[1]])
  if(is.null(nomi)){
    #if no name is specified, set it as 'theta' by default
    nomi <- base::paste0("theta",seq_along(theta0[[1]]))
  }
  
  
  #MCMC
  if(!parallel){
    
    #initialize the output list
    mcmc_out <- list(chains = list(),
                     d = length(theta0[[1]]),
                     k = k,
                     K = K,
                     N = N,
                     method = method,
                     tau = tau,
                     L = L,
                     thin = thin,
                     control = control,
                     verbose = verbose,
                     parallel = parallel)
    
    #create the seed for each cluster, this is obtain the same results
    #with the parallel case
    #seeds <- stats::rexp(n_chains,1e-3)
    
    #let's cycle for every chain
    for(i in seq_len(n_chains)){
      #set seed
      #set.seed(seeds[i])
      
      if(hide){
        #if the user doesn't want it printed on console
        #use the function quietly to silent it
        mcmc_out$chains[[i]] <- purrr::quietly(main_function)(theta0 = theta0[[i]],
                                              nlp = nlp,
                                              args = args,
                                              k = k,
                                              N = N,
                                              K = K,
                                              tau = tau,
                                              L = L,
                                              thin = thin,
                                              chain_id = i,
                                              verbose = verbose,
                                              control = control)$result
      }else{
        #otherwise use the plain main_function
        mcmc_out$chains[[i]] <- main_function(theta0 = theta0[[i]],
                                              nlp = nlp,
                                              args = args,
                                              k = k,
                                              N = N,
                                              K = K,
                                              tau = tau,
                                              L = L,
                                              thin = thin,
                                              chain_id = i,
                                              verbose = verbose,
                                              control = control)
      }
      
      #let's give the name of the parameters to each chain
      base::colnames(mcmc_out$chains[[i]]$values) <- nomi
      if(control$keep_warm_up == TRUE){
        #do the same on the warm up matrices if present
        base::colnames(mcmc_out$chains[[i]]$warm_up) <- nomi
      }
      if(!hide){
        base::cat("\n")
      }
    }
    
    #let's make the output an S3 object by assigning it a class
    class(mcmc_out) <- "XDNUTS"
    
  }else{
    #parallel chains
    #require(parallel)
    
    #create the seed for each cluster
    #seeds <- stats::rexp(n_chains,1e-3)
    
    #creation of the function to run in parallel
    #f <- function(i,theta0,nlp,args,k,N,K,tau,L,thin,verbose,control,seeds){
    f <- function(i,theta0,nlp,args,k,N,K,tau,L,thin,verbose,control){
      #set the seed of this cluster
      #set.seed(seeds[i])
      
      #call the C++ function
      main_function(theta0 = theta0[[i]],
                    nlp = nlp,
                    args = args,
                    k = k,
                    N = N,
                    K = K,
                    thin = thin,
                    tau = tau,
                    L = L,
                    chain_id = i,
                    verbose = verbose,
                    control = control)
    }
    
    #cluster initialization
    if(hide){
      #no output are printed to console
      cl <- parallel::makeCluster(n_chains)
    }else{
      #the output are printed to console
      cl <- parallel::makeCluster(n_chains, outfile = "")
    }
    
    #pass the libraries and object required to each core
    if(length(loadRObject) > 0 ){
      
      #pass object
      parallel::clusterExport(cl, loadRObject)
      
    }
    if(length(loadLibraries) > 0){
      
      #pass libraries names
      parallel::clusterExport(cl, "loadLibraries",envir = environment())
      
      #load libraries
      parallel::clusterEvalQ(cl, {
          eval(parse(text = paste0("library(",loadLibraries,")")))
      })
      
    }

    #parallel chains
    res <- base::tryCatch(parallel::parLapply(cl,
                          seq_len(n_chains),
                          f,
                          theta0 = theta0,
                          nlp = nlp,
                          args = args,
                          k = k,
                          N = N,
                          K = K,
                          tau = tau,
                          L = L,
                          thin = thin,
                          verbose = verbose,
                          control = control),#,
                          #seeds = seeds),
                    error = function(x) NULL)
    
    #stop the cluster
    parallel::stopCluster(cl)
    
    #verify that everything is ok, otherwise reports an error
    if(is.null(res)) base::stop("Something went wrong in processing the sampling in parallel. Try parallel = FALSE.")
    
    #let's assign each coordinate it's name, for every chain
    for(i in seq_along(res)){
      base::colnames(res[[i]]$values) <- nomi
      if(control$keep_warm_up == TRUE && control$N_adapt > 0){
        #do the same to the warm up matrices if present
        base::colnames(res[[i]]$warm_up) <- nomi
      }
    }
    
    #let's create the output list
    mcmc_out <- list(chains = res,
                     d = length(theta0[[1]]),
                     k = k,
                     K = K,
                     N = N,
                     method = method,
                     tau = tau,
                     L = L,
                     thin = thin,
                     control = control,
                     verbose = verbose,
                     parallel = parallel)
    
    #let's make the output an S3 object by assigning it a class
    class(mcmc_out) <- "XDNUTS"
    
  }
  
  #count the number of divergent transitions encountered during Hamilton equation
  #approximation
  n_div <- sum(base::sapply(mcmc_out$chains,function(x) sum(base::NROW(x$div_trans))))
  if(n_div != 0){
    #report to the user the number of this
    base::warning("\n",n_div, " trajectory ended with a divergent transition!\nConsider increasing 'control$delta' via 'set_parameters' to reduce bias.")
  }
  
  #let's make sure that the K field of the output list is the one relative
  #to the sampling phase and not the warm up one
  if(control$recycle_only_init){
    mcmc_out$K <- 1
  }
  
  #return the output
  return(mcmc_out)
}

### PRINT FUNCTION OF AN XDNUTS OBJECT
#' Function for printing an object of class XDNUTS
#' 
#' Print to console the specific of the algorithm used to generate an XDNUTS object.
#' 
#' @param x an object of class XDNUTS.
#' @param ... additional arguments to pass. These currently do nothing.
#' @param digits a numeric scalar indicating the number of digits to show. Default value is 3.
#' @param show_all logical scalar indicating where to print all the summary statistics 
#'  even if these are more than 10.
#'  
#' @return Return a graphical object.
#' 
#' @export print.XDNUTS
print.XDNUTS <- function(x,... , digits = 3, show_all = FALSE){
  
  #check inputs
  if(!base::inherits(x,"XDNUTS")){
    base::stop("'x' must an object of class 'XDNUTS'!")
  }
  
  if(!is.numeric(digits) || any(digits < 1) || length(digits) > 1){
    base::stop("'digits' must be a scalar positive numeric value!")
  }
  
  if(!is.logical(show_all) || length(show_all) > 1){
    base::stop("'show_all' must be a logical scalar value!")
  }
  
  #get number of chains
  nc <- length(x$chains)
  
  base::cat("\n")
  
  #print to console the type of algorithm employed
  if(x$parallel){
    
    #parallel chains
    if(nc > 1){
      
      #more than one chain
      base::cat(x$method,"algorithm on", nc, "parallel chains")
    }else{
      
      #only one chain
      base::cat(x$method,"algorithm on", nc, "chain")
    }
  }else{
    #chain not in parallel
    if(nc > 1){
      
      #mote than one chain
      base::cat(x$method,"algorithm on", nc, "chains")
    }else{
      
      #only one chain
      base::cat(x$method,"algorithm on", nc, "chain")
    }
  }
  
  #print method specifics
  if(x$method == "XHMC"){
    
    #XHMC
    base::cat(" with tau =", x$tau)
  }else if(x$method == "HMC"){
    
    #HMC
    base::cat(" with L =", x$L)
  }
  
  #print parameters specific
  base::cat("\nParameter dimension:",x$d)
  base::cat("\nNumber of discontinuous components:",x$k)
  
  #print stochastic optimization procedure specifics
  base::cat("\nStep size stochastic optimization procedure:")
  if(x$control$N_init1 > 0 || x$control$N_init2 > 0){
    
    #employed
    if(x$d > x$k){
      base::cat("\n - penalize deviations from nominal global acceptance rate")
    }
    if(x$control$different_stepsize && x$k > 0){
      
      #different step size
      base::cat("\n - different step size for each discontinuous component")
      
    }else if(!is.na(x$control$delta[2]) && x$k > 0){
      
      #one or more refraction rates?
      if(x$k > 1){
        
        #many
        base::cat("\n - penalize deviations from nominal refraction rates ")
      }else{
        
        #one
        base::cat("\n - penalize deviations from nominal refraction rate ")
      }
    }
  }else{
    #not employed
    base::cat("none")
  }
  base::cat("\n")
  
  
  #print samples specific
  base::cat("\nNumber of iteration:", x$N)
  base::cat("\nNumber of recycled samples from each iteration:",(x$K-1)*I(!x$control$recycle_only_init))
  base::cat("\nThinned samples:",x$thin)
  base::cat("\nTotal sample from the posterior:", base::NROW(x$chains[[1]]$values)*nc)
  
  #process output
  oo <- round(base::t(base::sapply(x$chains,function(y) 
    c(s1 = stats::median(y$energy), #median energy
      s2 = stats::median(y$delta_energy), #median delta energy
      s3 = stats::var(y$delta_energy)/stats::var(y$energy), #EBFMI
      s4 = stats::median(y$step_size), #median step size
      s5 = stats::median(y$step_length), #median step length
      s6 = y$alpha))),digits = digits) #adaption rates
  
  #fix cols and rows names
  base::rownames(oo) <- base::paste0("chain",seq_along(x$chains))
  base::colnames(oo)[1:5] <- c("Me(E)","Me(dE)","EBFMI","Me(eps)","Me(L)")
  
  #check for problematic samples
  n_div <- base::sapply(x$chains,function(y) sum(base::NROW(y$div_trans)))
  n_premature <-  base::sapply(x$chains,function(y) 
    sum(y$step_length == (2^x$control$max_treedepth - 1)))
  div_trans <- as.numeric(sum(n_div) > 0)
  premature <- as.numeric(sum(n_premature) > 0)
  
  if(div_trans){
    base::cat("\n")
    #divergent transitions
    base::message(sum(n_div), " trajectory ended with a divergent transition!\nConsider increasing 'control$delta' via 'set_parameters' to reduce bias.")
    oo <- base::cbind(oo[,1:5,drop = FALSE],n_div,oo[,-(1:5),drop = FALSE])
    base::colnames(oo)[6] <- "#divergence"
  }
  if(premature){
    base::cat("\n")
    #premature ending trajectory
    base::message(sum(n_premature), " trajectory ended before reaching an effective termination!\nFlat regions, consider increasing 'control$max_treedepth' via 'set_controls'.")
    oo <- base::cbind(oo[,1:(5+div_trans),drop = FALSE],n_premature,oo[,-(1:(5+div_trans)), drop = FALSE])
    base::colnames(oo)[6+div_trans] <- "#premature"
  }
  
  #print chains statistics
  if(nc > 1){
    base::cat("\nChains statistics:\n")
  }else{
    base::cat("\nChain statistics:\n")
  }
  
  #legend
  base::cat("\n - E: energy of the system")
  base::cat("\n - dE: energy first differce")
  base::cat("\n - EBFMI = Var(E)/Var(dE): Empirical Bayesian Fraction of Missing Information\n   A value lower than 0.2 it is a sign of sub optimal momenta distribution.")
  base::cat("\n - eps: step size of the algorithm")
  base::cat("\n - L: number of step done per iteration")
  
  if(x$k == x$d){
    
    #full discontinuous: only refraction rates
    if(x$k > 1){
      
      #many
      base::cat("\n - alphas: refraction rates")
    }else{
      
      #only one
      base::cat("\n - alphas: refraction rate")
    }
    
    #fix colnames
    base::colnames(oo)[-(1:(5 + div_trans + premature))] <- base::paste0("alpha",seq_len(x$d))
  }else{
    #both global and refraction rates
    
    #global
    base::cat("\n - alpha0: global acceptance rate")
    
    #if there are refraction rates
    if(x$k > 1){
      
      #many
      base::cat("\n - alphas: refraction rates")
    }else{
      
      #only one
      base::cat("\n - alpha1: refraction rate")
    }
    
    #fix colnames
    base::colnames(oo)[-(1:(5 + div_trans + premature))] <- base::paste0("alpha",0:x$k)
  }
  
  base::cat("\n\n")
  
  #print statistics
  if(show_all || base::NCOL(oo) <= 10){
    #show all
    base::print(oo)
  }else{
    #show only the first 10
    base::print(oo[,1:10,drop = FALSE])
    base::cat("\nOmitting",base::NCOL(oo) - 10, "columns, set 'show_all = TRUE' to see all of them.")
  }
  
  #get alpha0
  alpha0 <- oo[,6+div_trans+premature]
  
  #get other alphas
  alphas <- base::t(oo[,-(1:(5+div_trans+premature)),drop = FALSE])
  
  #fix the case without global rate
  if(x$k == x$d){
    #set it to the nominal value
    alpha0 <- rep(x$control$delta[1],length(x$chains))
  }else{
    #drop the global rate
    alphas <- alphas[-1,,drop = FALSE]
  }
  
  #get nominal values
  deltas <- rep(x$control$delta,c(1,x$k))
  
  #impute missing values with the default
  deltas <- base::ifelse(is.na(deltas),0.6,deltas)
  
  #calculate the mean absolute differences
  mae0 <- alpha0 - deltas[1]
  if(x$k == 0){
    mae1 <- 0
  }else{
    mae1 <- base::apply(abs(alphas - deltas[-1]),2,base::mean)
  }

  #compute the number of problematic chains
  n0_min <- sum(mae0 < -0.1)
  n0_max <- sum(mae0 > 0.19)
  n1 <- sum(mae1 > 0.15)
  
  if(n0_min == 1){
    base::message("\n\n1 chain has a global acceptance rate that is significantly lower than the nominal value!")
  }else if(n0_min > 1){
    base::message("\n\n,",n0_min," chains has a global acceptance rate that are significantly lower than the nominal value!")
  }
  if(n1 == 1){
    if(x$k == 1){
      base::message("\n\n1 chain has a refraction rate that is significantly different from the nominal value!")
    }else{
      base::message("\n\n1 chain has refraction rates that are significantly different from the nominal values!")
    }
  }else if(n1 > 1){
    if(x$k == 1){
      base::message("\n\n",n1," chains has a refraction rate that is significantly different from the nominal value!")
    }else{
      base::message("\n\n",n1," chains has refraction rates that are significantly different from the nominal values!")
    }
  }
  if((n0_min > 0 || n0_max > 0 || n1 > 0) && x$k > 1){
    base::message("Consider setting 'different_stepsize = ",!x$control$different_stepsize,"', or proceed with manual tuning")
  }
  
  base::cat("\n")
  
  #return statistics
  invisible(oo)
  
}


### PLOTS FUNCTION OF THE MCMC OUTPUT
#' Function to view the draws from the posterior distribution.
#'
#' @param x an object of class \code{XDNUTS}.
#' @param type the type of plot to display. \describe{
#' \item{\code{type = 1}}{marginal chains, one for each desired dimension.}
#' \item{\code{type = 2}}{bivariate plot.}
#' \item{\code{type = 3}}{time series plot of the energy level sets. Good for a quick diagnostic of big models.}
#' \item{\code{type = 4}}{stickplot of the step-length of each iteration.}
#' \item{\code{type = 5}}{Histograms of the centered marginal energy in gray and of the first differences of energy in red.}
#' \item{\code{type = 6}}{Autoregressive function plot of the parameters.}
#' \item{\code{type = 7}}{Matplot of the empirical acceptance rate and refraction rates.}}
#'
#' @param which either a numerical vector indicating the index of the parameters of interest or a string \describe{
#' \item{\code{which = 'continuous'}}{for plotting the first \eqn{d-k} parameters.}
#' \item{\code{which = 'discontinuous'}}{for plotting the last \eqn{k} parameters.}
#' }
#' where both \eqn{d} and \eqn{k} are elements contained in the output of the \link{xdnuts} function.
#'  If \code{type = 7}, it refers to the rates index instead. When \code{which = 'continuous'}, 
#'  only the global acceptance rate is displayed. In contrast, when \code{which = 'discontinuous'},
#'   the refraction rates are shown.
#' @param warm_up a boolean value indicating whether the plot should be made using the warm-up samples.
#' @param plot.new a boolean value indicating whether a new graphical window should be opened. This is advised if the parameters space is big.
#' @param which_chains a numerical vector indicating the index of the chains of interest.
#' @param colors a numerical vector containing the colors for each chain specified in \code{which_chains}
#'  or for each rate specified in \code{which} when \code{type = 7}.
#' @param gg A boolean value indicating whether the plot should utilize the grammar of graphics features. 
#' Default value is set to \code{TRUE}.
#' @param scale A numeric value for scaling the appearance of plots.
#' @param ... additional arguments to customize plots. In reality, these do nothing.
#' 
#' @return A graphical object if \code{gg = TRUE}, otherwise nothing is returned.
#'
#' @export plot.XDNUTS
#' @export
plot.XDNUTS <- function(x,type = 1,which = NULL,warm_up = FALSE,
                       plot.new = FALSE, which_chains = NULL,colors = NULL,
                       gg = TRUE, scale = 1,...){
  
  #check inputs
  if(!base::inherits(x,"XDNUTS")){
    base::stop("'x' must be an object of class 'XDNUTS'!")
  }
  
  if(!is.logical(warm_up) || length(warm_up) > 1){
    base::stop("'warm_up' must be a logical scalar value!")
  }
  
  if(!is.logical(plot.new) || length(plot.new) > 1){
    base::stop("'warm_up' must be a logical scalar value!")
  }
  
  if(!base::is.logical(gg) || length(gg) > 1){
    base::stop("''gg' must be a logical scalar value!")
  }
  
  #if(!is.numeric(1) || length(1) > 1){
  #  base::stop("'1' must be a scalar numeric value!")
  #}
  
  #if(!is.numeric(1) || length(1) > 1){
  #  base::stop("'1' must be a scalar numeric value!")
  #}
  
  #save current graphic window appearence in order to reset them at the end
  #op <- graphics::par(no.readonly = TRUE)
  
  #reset the graphic window on.exit
  #on.exit(graphics::par(op))
  
  #get the number of chains
  nc <- length(x$chains)
  
  #initialize and make sure that the index of the chain to use is admissible
  if(is.null(which_chains)){
    which_chains <- seq_len(nc)
  }else{
    if(any(which_chains > nc | which_chains < 1)){
      base::stop("Incorrect chain indexes!")
    }
    which_chains <- base::unique(which_chains)
  }
  
  #which parameters/rates do we want to see the graph of?
  #make sure it the input is admissible
  if(is.null(which)){
    if(type == 7){
      #rates
      which <- seq_along(x$chains[[1]]$alpha)
    }else{
      #parameters
      which <- seq_len(base::NCOL(x$chains[[1]]$values))
    }
  }else if(all(which == "continuous")){
    
    if(type == 7){
      #rates
      which <- base::ifelse(x$k == x$d,base::integer(0),1)
    }else{
      #parameters
      which <- seq_len(x$d)[seq_len(x$d-x$k)]
    }

  }else if(all(which == "discontinuous")){
    
    if(type == 7){
      #rates
      which <- base::ifelse(x$k == x$d,seq_len(x$k),1+seq_len(x$k))
    }else{
      #parameters
      which <- seq_len(x$d)[-seq_len(x$d-x$k)]
    }
    
  }else if(any(which < 1) || 
           any(which > base::NCOL(x$chains[[1]]$values) & type != 7)  ||
           any(which > length(x$chains[[1]]$alpha) & type == 7)     ){
    base::stop("Incorrect index of parameters!")
  }
  
  #do the same with the colors argument
  if(is.null(colors)){
    if(type == 7){
      #rates
      colori <- base::sapply(base::seq_along(which),grDevices::adjustcolor,alpha = 0.5)
    }else{
      #parameters
      colori <- base::sapply(seq_len(nc),grDevices::adjustcolor,alpha = 0.5)
    }
  }else{
    if(type == 7){
      #rates
      colori <- base::cbind(colors,base::seq_along(which))[,1]
    }else{
      #parameters
      colori <- base::cbind(colors,1:nc)[,1]
    }
  }

  #do we want to see the warm up or the sampling?
  #make sure that the input is admissible
  if(warm_up == TRUE){
    quale <- "warm_up"
    if(is.null(x$chains[[1]][[quale]])){
      base::stop("No warm-up phase available!")
    }
  }else{
    quale <- "values"
  }
  
  #do we want a new graphics window?
  if(plot.new) {
    grDevices::X11()
  }
  
  if(gg){
    #GRAMMAR OF GRAPHICS
    
    #set to null all the possible variable names
    Chain <- Index <- Value <- Freq <- Var1 <- Var2 <- Type <- Var <- Par <- Color <- NULL
    
    #plot1: marginal chains
    if(type == 1){
      
      #get number of rows
      nr <- NROW(x$chains[[1]][[quale]])
      
      #get variables names
      nomi <- base::colnames(x$chains[[1]]$values)[which]
      
      #create the data.frame for ggplot2
      df <- base::do.call(base::rbind,base::lapply(which_chains, function(i) 
        base::data.frame(Value = c(x$chains[[i]][[quale]][,which]),
                         Index = seq_len(nr),
                         Var = rep(nomi,each = nr),
                         Chain = i) ))
      
      #convert chain index in factor
      df$Chain <- base::as.factor(df$Chain)
      
      #create the graphic
      G <- ggplot2::ggplot(df, ggplot2::aes(x = Index, y = Value, color = Chain)) +
        ggplot2::geom_line(linewidth = 0.1) +  #add trace lines
        ggplot2::scale_color_manual(values = colori) +  # paletta
        ggplot2::guides(color = ggplot2::guide_legend( 
          override.aes=list(linewidth = 1, alpha = 0.8))) + #make the legend line bigger
        ggplot2::labs(title = NULL, x = NULL, y = NULL) + #add title
        ggplot2::theme_gray() + #add theme
        ggplot2::theme(
          legend.position = "right", #legend to the right
          legend.justification = "top", #legend on top
          plot.title = ggplot2::element_text(hjust = 0.5),  #center the title
          axis.text.x = ggplot2::element_blank(),  #remove x axis
          axis.title.x = ggplot2::element_blank(),
          panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
          panel.grid.minor.x = ggplot2::element_blank()) + 
        ggplot2::facet_wrap(~ Var, scales = "free_y")  # make a trace plot for every variable
      
      #return the plot
      return(G)
      
    }else if(type == 2){
      #plot2: marginal and bivariate densities
      
      #create the data frame for ggplot2
      
      #get number of rows
      nr <- NROW(x$chains[[1]][[quale]])
      
      #get variables names
      nomi <- base::colnames(x$chains[[1]]$values)[which]
      
      #create the data.frame for ggplot2
      df <- base::do.call(base::rbind,base::lapply(which_chains, function(i) 
        base::data.frame(x$chains[[i]][[quale]][,which,drop = FALSE],
                         Chain = i) ))
      
      #convert chain index in factor
      df$Chain <- base::as.factor(df$Chain)
      
      #get dimension
      d <- length(nomi)
      
      # GRIDEXTRA VERSION
      
      #create the layout matrix
      lmat <- base::cbind(base::sapply(rep(d*(1:d-1),each = 2),function(x) 
        x + rep(seq_len(d),each = 2)),d*d+1)
      
      #create the list of plot objects
      ll <- base::vector("list",d*d + 1)
      
      #save x_axis ranges
      xy_range <- base::matrix(NA,2,d)
      
      #add diagonal plots one at the time
      for(i in seq_len(d)){
        #create the sub plot
        tmp <- ggplot2::ggplot(base::data.frame(Value = df[,nomi[i]],Chain = df$Chain),
                               ggplot2::aes(x = Value, color = Chain)) +
          ggplot2::geom_density(alpha = 0.5, linewidth = 0.5, show.legend = FALSE) + 
          ggplot2::scale_color_manual(values = colori) +  # paletta
          ggplot2::labs(title = nomi[i], x = NULL, y = NULL) + 
          ggplot2::theme_gray() + 
          ggplot2::theme(
            panel.grid.major.y = ggplot2::element_blank(),  # remove horizontal grid
            panel.grid.minor.y = ggplot2::element_blank(),
            plot.title = ggplot2::element_text(hjust = 0.5, face = "bold")
          )
        
        #insert the plot on the right spot on the list
        ll[[d*(i-1)+i]] <- tmp
        
        #save x_axis limits
        xy_range[,i] <- ggplot2::layer_scales(tmp)$x$range$range
      }
      
      #add upper diagonal plots one at the time
      for(i in seq_along(nomi)){
        for(j in base::setdiff(seq_along(nomi),seq_len(i))){
          #create the subplot
          tmp <- ggplot2::ggplot(base::data.frame(Var1 = df[,nomi[j]],
                                                  Var2 = df[,nomi[i]],
                                                  Color = grDevices::densCols(df[,nomi[i]], df[,nomi[j]],
                                                                              colramp = grDevices::colorRampPalette(c("gray90", grDevices::blues9)) ))) + 
            ggplot2::xlim(xy_range[1,j],xy_range[2,j]) + 
            ggplot2::ylim(xy_range[1,i],xy_range[2,i]) + 
            ggplot2::geom_point(ggplot2::aes(x = Var1,y =  Var2, col = Color)) + 
            ggplot2::scale_color_identity() +
            ggplot2::labs(title = "", x = NULL, y = NULL) + 
            ggplot2::theme_gray() + 
            ggplot2::theme(
              panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
              panel.grid.minor.x = ggplot2::element_blank(),
              panel.grid.major.y = ggplot2::element_blank(),  # remove horizontal grid
              panel.grid.minor.y = ggplot2::element_blank()
            )
          
          #insert the plot in the right spot on the list
          ll[[ d*(j-1) + i ]] <- tmp
        }
      }
      
      #create the legend
      tmp <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(
        ggplot2::ggplot(base::data.frame(Value = df[,nomi[1]],Chain = df$Chain),
                        ggplot2::aes(x = Value, color = Chain)) +
          ggplot2::scale_color_manual(values = colori) +  
          ggplot2::stat_density(alpha = 0.5, geom = "line") + 
          ggplot2::guides(color = ggplot2::guide_legend( 
            override.aes=list(linewidth = 2)))))
      
      #insert it in the right spot
      ll[[d*d + 1]] <- tmp$grobs[[base::which(base::sapply(tmp$grobs, function(x) x$name) == "guide-box")]]
      
      #create the plot
      graphics::plot.new()
      gridExtra::grid.arrange(gridExtra::arrangeGrob(grobs = ll,
                                                     layout_matrix = lmat),
                              newpage = FALSE)
      #save the plot
      G <- grDevices::recordPlot()
      
      #return the plot
      return(G)
      
      # GRID VERSION
      
      # #create the empty grid
      # grid::grid.newpage()
      # grid::pushViewport(grid::viewport(layout = grid::grid.layout(2*d, 2*d+1)))
      # 
      # #save x_axis ranges
      # xy_range <- base::matrix(NA,2,d)
      # 
      # #add diagonal plots one at the time
      # for(i in seq_len(d)){
      #   #create the sub plot
      #   tmp <- ggplot2::ggplot(base::data.frame(Value = df[,nomi[i]],Chain = df$Chain),
      #                          ggplot2::aes(x = Value, color = Chain)) +
      #     ggplot2::geom_density(alpha = 0.5, show.legend = FALSE) + 
      #     ggplot2::labs(title = nomi[i], x = NULL, y = NULL) + 
      #     ggplot2::theme_gray() + 
      #     ggplot2::theme(
      #       panel.grid.major.y = ggplot2::element_blank(),  # remove horizontal grid
      #       panel.grid.minor.y = ggplot2::element_blank(),
      #       plot.title = ggplot2::element_text(hjust = 0.5, face = "bold")
      #     )
      #   
      #   #add the sub plot
      #   grid::pushViewport(grid::viewport(layout.pos.col = 2*(i-1)+1:2, layout.pos.row = 2*(i-1)+1:2))
      #   base::print(tmp, newpage = FALSE)
      #   grid::popViewport(1)
      #   
      #   #save x_axis limits
      #   xy_range[,i] <- ggplot2::layer_scales(tmp)$x$range$range
      # }
      # 
      # #add upper diagonal plots one at the time
      # for(i in seq_along(nomi)){
      #   for(j in base::setdiff(seq_along(nomi),seq_len(i))){
      #     #create the subplot
      #     tmp <- ggplot2::ggplot(base::data.frame(Var1 = df[,nomi[j]],
      #                                             Var2 = df[,nomi[i]],
      #                                             Color = grDevices::densCols(df[,nomi[i]], df[,nomi[j]],
      #                                                                         colramp = grDevices::colorRampPalette(c("gray90", grDevices::blues9)) ))) + 
      #       ggplot2::xlim(xy_range[1,j],xy_range[2,j]) + 
      #       ggplot2::ylim(xy_range[1,i],xy_range[2,i]) + 
      #       ggplot2::geom_point(ggplot2::aes(x = Var1,y =  Var2, col = Color)) + 
      #       ggplot2::scale_color_identity() +
      #       ggplot2::labs(title = "", x = NULL, y = NULL) + 
      #       ggplot2::theme_gray() + 
      #       ggplot2::theme(
      #         panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
      #         panel.grid.minor.x = ggplot2::element_blank(),
      #         panel.grid.major.y = ggplot2::element_blank(),  # remove horizontal grid
      #         panel.grid.minor.y = ggplot2::element_blank()
      #       )
      #     
      #     #add the sub plot
      #     grid::pushViewport(grid::viewport(layout.pos.row = 2*(i-1)+1:2, layout.pos.col = 2*(j-1)+1:2))
      #     base::print(tmp, newpage = FALSE)
      #     grid::popViewport(1)
      #     
      #   }
      # }
      # 
      # #create the legend
      # tmp <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(
      #   ggplot2::ggplot(base::data.frame(Value = df[1,nomi[1]],Chain = df$Chain),
      #                   ggplot2::aes(x = Value, color = Chain)) +
      #     ggplot2::stat_density(alpha = 0.5, geom = "line")))
      # legend <- tmp$grobs[[base::which(base::sapply(tmp$grobs, function(x) x$name) == "guide-box")]]
      # 
      # #add the legend
      # grid::pushViewport(grid::viewport(layout.pos.col = 2*d+1, layout.pos.row = seq_len(2*d)))
      # grid::grid.draw(legend)
      # grid::popViewport(0)
      
      # GGALLY VERSION
      
      # #define the empty plot with GGally
      # G <- GGally::ggpairs(df,columns = seq_along(nomi), upper='blank', diag='blank', lower='blank',
      #                      legend = c(1,1))
      # 
      # #save x_axis ranges
      # xy_range <- base::matrix(NA,2,length(nomi))
      # 
      # #add diagonal plots one at the time
      # for(i in seq_along(nomi)){
      #   #create the sub plot
      #   tmp <- ggplot2::ggplot(base::data.frame(Value = df[,nomi[i]],Chain = df$Chain),
      #                          ggplot2::aes(x = Value, color = Chain)) +
      #     ggplot2::geom_density(alpha = 0.5, show.legend = FALSE) + 
      #     ggplot2::labs(title = nomi[i], x = NULL, y = NULL) + 
      #     ggplot2::theme_gray() + 
      #     ggplot2::theme(
      #       panel.grid.major.y = ggplot2::element_blank(),  # remove horizontal grid
      #       panel.grid.minor.y = ggplot2::element_blank(),
      #       plot.title = ggplot2::element_text(hjust = 0.5)
      #       )
      #   #add the sub plot
      #   G <- GGally::putPlot(G, tmp, i, i)
      #   
      #   #save x_axis limits
      #   xy_range[,i] <- ggplot2::layer_scales(tmp)$x$range$range
      # }
      # 
      # #add upper diagonal plots one at the time
      # for(i in seq_along(nomi)){
      #   for(j in base::setdiff(seq_along(nomi),seq_len(i))){
      #     #create the subplot
      #     tmp <- ggplot2::ggplot(base::data.frame(Var1 = df[,nomi[i]],
      #                                             Var2 = df[,nomi[j]],
      #                                             Color = grDevices::densCols(df[,nomi[i]], df[,nomi[j]],
      #                                                                         colramp = grDevices::colorRampPalette(c("gray90", grDevices::blues9)) ))) + 
      #       ggplot2::xlim(xy_range[1,i],xy_range[2,i]) + 
      #       ggplot2::ylim(xy_range[1,j],xy_range[2,j]) + 
      #       ggplot2::geom_point(ggplot2::aes(x = Var1,y =  Var2, col = Color)) + 
      #       ggplot2::scale_color_identity() +
      #       ggplot2::theme_gray() + 
      #       ggplot2::theme(
      #         panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
      #         panel.grid.minor.x = ggplot2::element_blank(),
      #         panel.grid.major.y = ggplot2::element_blank(),  # remove horizontal grid
      #         panel.grid.minor.y = ggplot2::element_blank()
      #       )
      #       
      #     #add the sub plot
      #     G <- GGally::putPlot(G, tmp, i, j)
      #   }
      # }
      # 
      # #return the plot
      # return(G)
      
    }else if(type == 3){
      #plot3: energy Markov chain
      
      #get length
      n <- length(x$chains[[1]]$energy)
      
      #gets each chain energy
      df <- base::do.call(base::rbind,base::lapply(which_chains,function(i) 
        base::data.frame(Value = x$chains[[i]]$energy,
                         Index = seq_len(n),
                         Chain = i)))
      
      #make Chain of type factor
      df$Chain <- base::as.factor(df$Chain)
      
      #create the graphical object
      G <- ggplot2::ggplot(df,ggplot2::aes(x = Index, y = Value, color = Chain)) + 
        ggplot2::geom_line(linewidth = 0.2) +  #add trace lines
        ggplot2::scale_color_manual(values = colori) +  #palette
        ggplot2::guides(color = ggplot2::guide_legend( 
          override.aes=list(linewidth = 1, alpha = 0.8))) + #make the legend line bigger
        ggplot2::labs(title = NULL, x = "Iteration", y = "Energy") + #add labels
        ggplot2::theme_gray() + #add theme
        ggplot2::theme(
          plot.title = ggplot2::element_text(hjust = 0.5),  # center title
          legend.position = "right",  # Legenda a destra
          legend.justification = "top", #legend on top
          panel.grid.major.x = ggplot2::element_blank(),  #delete vertical grid
          panel.grid.minor.x = ggplot2::element_blank()
        )
      
      #return the plot
      return(G)
      
    }else if(type == 4){
      #plot4: stick plot of trajectory length
      
      #get the frequencies of step lengths for each chain
      df <- base::do.call(base::rbind,base::lapply(which_chains, function(i) {
        tmp <- base::table(x$chains[[i]]$step_length)
        base::data.frame(Var1 = base::as.numeric(base::names(tmp)),
                         Freq = base::as.numeric(tmp),Chain = i)
      }))
      
      #set Chain id to factor
      df$Chain <- base::as.factor(df$Chain)
      
      #create the grid
      
      #aggregate the frequencies
      breaks <- base::as.numeric(
        base::names(
          base::sort(base::tapply(
            df$Freq,df$Var1,base::mean),decreasing = TRUE)))
      
      breaks <- base::unique(
        base::sort(
          c(breaks[1:10],base::range(breaks)),decreasing = FALSE))
      
      #create the graphical object
      G <- ggplot2::ggplot(df,ggplot2::aes(x = Var1, y = Freq)) + 
        ggplot2::geom_segment(ggplot2::aes(xend = Var1, yend = 0, color = Chain),
                              position = ggplot2::position_jitter(width = 0.2)) +
        ggplot2::scale_x_continuous(breaks = breaks) + 
        ggplot2::scale_color_manual(values = colori) +  #palette
        ggplot2::guides(color = ggplot2::guide_legend( 
          override.aes=list(linewidth = 1, alpha = 0.8))) + #make the legend line bigger
        ggplot2::labs(title = "Iterarion's Step Length", 
                      x = "L", y = "Frequency", color = "Chain") + 
        ggplot2::theme_gray() +  
        ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
      
      #return the graphical object
      return(G)
      
    }else if(type == 5){
      
      #plot5: marginal and first difference energy histogram
      
      #get energy chains
      E <- base::sapply(which_chains,function(i) x$chains[[i]]$energy)
      
      #center it
      E <- c(E) - stats::median(E)
      
      #get first difference energy chains
      delta_E <- c(base::sapply(which_chains,function(i) x$chains[[i]]$delta_energy))
      
      #get sample size
      N <- prod(dim(E))
      
      #create the data frame
      df <- base::data.frame(Value = c(E,delta_E),
                             Type = factor(rep(c("E","dE"),each = N),levels = c("E","dE")))
      
      #set Delta to NULL for avoiding stupid CRAN problems
      Delta <- NULL
      
      #create the graphical object
      G <- ggplot2::ggplot(df, ggplot2::aes(x = Value, fill = Type)) +
        ggplot2::geom_density(alpha = 0.5, color = NA) +  #energy densities
        ggplot2::scale_fill_manual( #legend
          values = c("E" = "dimgray", "dE" = "coral3"), 
          labels = c("E",base::expression(Delta * E) )) + 
        ggplot2::labs(title = NULL,  #labels
                      x = "Centered Energy", 
                      y = "Density", 
                      fill = NULL) +  
        ggplot2::theme_gray()  + #theme
        ggplot2::theme(legend.position = "right")  # legend on the right
      
      #return the plot
      return(G)
      
      
    }else if(type == 6){
      #plot6: autocorrelation plots
      
      #get lag maximum number 
      lag_max <- floor(10 * log10(base::NROW(x$chains[[1]]$values)))
      
      #get variables names
      nomi <- base::colnames(x$chains[[1]]$values)[which]
      
      #create the data.frame for ggplot2
      df <- base::do.call(base::rbind,base::lapply(which_chains, function(i) 
        base::data.frame(Value = c(base::apply(x$chains[[i]][[quale]][,which,drop = FALSE],2,
                                               function(y) stats::acf(y, plot = FALSE, lax.max = lag_max)$acf[-1,,1])),
                         Index = seq_len(lag_max),
                         Var = rep(nomi,each = lag_max),
                         Chain = i) ))
      
      #if there are NAs report it
      idx_na <- base::which(is.na(df$Value))
      if(length(idx_na) > 0){
        base::message("NA values, probably due to some reducible chain.\nThe corresponding ACFs are not well-defined so these are not plotted.")
        df <- df[-idx_na,]
      }
      
      if(base::NROW(df) == 0){
        base::stop("\nNo valid chain!")
      }
      
      #convert chain index in factor
      df$Chain <- base::as.factor(df$Chain)
      
      #create the graphic
      G <- ggplot2::ggplot(df, ggplot2::aes(x = Index, y = Value, color = Chain)) +
        ggplot2::geom_line(linewidth = 0.2) +  #add trace lines
        ggplot2::scale_color_manual(values = colori) +  #palette
        ggplot2::guides(color = ggplot2::guide_legend( 
          override.aes=list(linewidth = 1, alpha = 0.8))) + #make the legend line bigger
        ggplot2::ylim(NA,1) + #set the upper limit to 1
        ggplot2::labs(title = "Autocorrelation Plots", x = NULL, y = NULL) + #add title
        ggplot2::theme_gray() + #add theme
        ggplot2::theme(
          legend.position = "right", #legend to the right
          legend.justification = "top", #legend on top
          plot.title = ggplot2::element_text(hjust = 0.5),  #center the title
          panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
          panel.grid.minor.x = ggplot2::element_blank()) + 
        ggplot2::facet_wrap(~ Var, scales = "free_y")  # make a trace plot for every variable
      
      #return the plot
      return(G)
      
    }else if(type == 7){
      #plot7: matplot of alphas
      
      #check the number of chains
      if(length(which_chains) == 1){
        stop("alphas plots has no sense with only one chain!")
      }
      
      #consider all the possible cases
      if(x$k == 0 || (x$d > x$k && all(which == 1))){
        #only the global empirical rate
        
        titolo <- "Empirical Global Acceptance Rate"
        nomi <- "0"
      }else if(x$k == x$d || (x$d > x$k && all(which != 1))){
        #only the refraction rates
        if(length(which) == 1){
          #only one
          
          titolo <- "Empirical Refraction Rate"
          nomi <- base::as.character(which)
        }else{
          #many
          
          titolo <- "Empirical Refraction Rates" 
          nomi <- base::as.character(which)
        }
      }else{
        #both the refraction and global rates
        
        titolo <- "Empirical Rates"
        nomi <- base::as.character(which-1)
      }
      
      #create the data frame for ggplot2
      df <- base::data.frame(Value = c(base::sapply(which_chains,function(i) x$chains[[i]]$alpha[which])),
                             Index = rep(which_chains,each = length(nomi)),
                             Par = nomi)
      
      #add the type of rate
      df$Type <- c("Global","Refraction")[2 - (df$Par == 0)]
      
      #create the graphical object                 
      G <- ggplot2::ggplot(df,ggplot2::aes(x = Index, y = Value)) + ggplot2::ylim(0,1) #set the y limits
      
      if(x$k == 0 || length(which) == 1 ){
        #no legend
        G <- G + ggplot2::geom_line(ggplot2::aes(color = Par, linetype = Type), show.legend = FALSE) + #add lines
          ggplot2::geom_point(ggplot2::aes(color = Par),show.legend = FALSE) +  #add points
          ggplot2::scale_color_manual(values = colori) +  #palette
          ggplot2::labs(title = titolo, x = "Chain", y = "Rate", #add labels
                        color = NULL,linetype = NULL) + 
          ggplot2::theme_gray() +  #add theme
          ggplot2::theme(
            plot.title = ggplot2::element_text(hjust = 0.5),  #center the title
            panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
            panel.grid.minor.x = ggplot2::element_blank())
      }else if(x$k > 0 && x$k < x$d && any(which == 1)){
        #use different linetype
        G <- G + ggplot2::geom_line(ggplot2::aes(color = Par, linetype = Type)) + #add lines
          ggplot2::geom_point(ggplot2::aes(color = Par)) +  #add points
          ggplot2::scale_color_manual(values = colori) +  #palette
          ggplot2::labs(title = titolo, x = "Chain", y = "Rate", #add labels
                        color = NULL,linetype = "Type") + 
          ggplot2::theme_gray() +  #add theme
          ggplot2::theme(
            legend.position = "right", #legend to the right
            legend.justification = "top", #legend on top
            plot.title = ggplot2::element_text(hjust = 0.5),  #center the title
            panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
            panel.grid.minor.x = ggplot2::element_blank())
      }else{
        #use the same linetype
        G <- G + ggplot2::geom_line(ggplot2::aes(color = Par)) + #add lines
          ggplot2::geom_point(ggplot2::aes(color = Par)) +  #add points
          ggplot2::scale_color_manual(values = colori) +  #palette
          ggplot2::labs(title = titolo, x = "Chain", y = "Rate", #add labels
                        color = NULL) + 
          ggplot2::theme_gray() +  #add theme
          ggplot2::theme(
            legend.position = "right", #legend to the right
            legend.justification = "top", #legend on top
            plot.title = ggplot2::element_text(hjust = 0.5),  #center the title
            panel.grid.major.x = ggplot2::element_blank(),  # remove vertical grid
            panel.grid.minor.x = ggplot2::element_blank())
      }
      
      #return the plot
      return(G)
    }else{
      base::message("No such plot, 'type' argument accept value between 1 and 7!")
      return(NULL)
    }
  }else{
    #BASE R PLOTS
    
    #update the dimansion of the parameter space to be plotted
    d <- length(which)
    
    #plot1: marginal chains
    if(type == 1){
      
      #get MCMC iteration
      tt <- seq_len(base::NROW(x$chains[[1]][[quale]]))
      
      #let's partition the graphics window accordingly
      k1 <- ceiling(sqrt(d))
      if(d <= k1*(k1-1)) {
        k2 <- k1 - 1
      }else{
        k2 <- k1
      }
      
      graphics::layout(mat = base::cbind(base::matrix(1:(k1*k2),k2,k1, byrow = TRUE),k1*k2+1))
      
      #set the margins
      graphics::par(mar = c(1.1,3.1,3.1,1.1), cex = scale, lwd = scale)
      
      #get iteration range
      xlim <- range(tt)
      
      #make a plot for every parameter
      for(i in which){
        
        #compute the range of this parameter chains
        ylim <- range(base::sapply(which_chains, function(idx) range(x$chains[[idx]][[quale]][,i])))
        
        #empty plot
        base::plot(NULL,xlim = xlim,ylim = ylim, xlab = "", ylab = "",
                   main = base::colnames(x$chains[[1]][[quale]])[i], cex.main = 1)
        
        #add each chain
        for(j in which_chains){
          graphics::lines(tt,x$chains[[j]][[quale]][,i], col = colori[j])
        }
        
        #add a dashed line in zero
        graphics::abline(h = 0, col = 2, lty = 2)
      }
      
      #empty plots due to a non perfect graphical window partition
      for(i in seq_len(k2*k1 - d)){
        graphics::plot.new()
      }
      
      #add legend on the right window
      graphics::par(mar = c(1.1,1.1,1.1,1.1), cex = scale, lwd = scale)
      base::plot(1, type = "n", axes = FALSE, ylab = "", xlab = "", bty = "n")
      graphics::legend("top", title = "Chain", legend = which_chains, col = colori[which_chains],
                       bty = "n", lty = 1, lwd = 2, cex = 1, title.cex = 1,bg = "transparent")
      
      
    }else if(type == 2){
      #plot2: marginal and bivariate densities
      
      #let's partition the graphics window accordingly
      graphics::par(mfrow = c(d,d), mar = c(0.6,0.6,0.6,0.6), cex = scale, lwd = scale)
      for(i in 1:d){
        for(j in 1:d){
          if(i > j) {
            #makes empty plots below the diagonal block
            graphics::plot.new()
          }
          if(i == j){
            #plots marginal densities for each chain on the diagonal block
            
            #get posterior densities for each chain of this parameter
            dens_out <- base::lapply(which_chains,function(idx){
              base::do.call( base::cbind, stats::density(x$chains[[idx]][[quale]][,which[i]])[c("x","y")])
            })
            
            #gets the x and y limits
            xlim <- range(base::sapply(seq_along(dens_out),function(ii) range(dens_out[[ii]][,1])))
            ylim <- range(base::sapply(seq_along(dens_out),function(ii) range(dens_out[[ii]][,2])))
            
            #empty plot
            base::plot(NULL,xlim = xlim,ylim = ylim,xlab = "",ylab = "",
                       main = base::colnames(x$chains[[1]][[quale]])[which[i]],
                       cex.main = 1)
            
            #overlap the density of each chain
            for(ii in seq_along(dens_out)){
              graphics::lines(dens_out[[ii]], col = colori[which_chains[[ii]]])
            }
            
            #add a vertical dashed line in zero
            graphics::abline(v = 0, col = 2, lty = 2)
          }
          if(i < j){
            #on the upper triangle section plot the bivariate density
            graphics::smoothScatter(base::do.call(base::rbind,base::lapply(which_chains,
                                                                           function(idx) x$chains[[idx]][[quale]][,which[c(j,i)]])), main = "")
            #add verttical and horizontal dashed line in zero
            graphics::abline(h = 0, col = 2, lty = 2)
            graphics::abline(v = 0, col = 2, lty = 2)
          }
        }
      }
    }else if(type == 3){
      #plot3: energy Markov chain
      
      #gets each chain energy
      energie <- base::sapply(which_chains,function(idx) x$chains[[idx]]$energy)
      
      #gets x and y limit
      xlim <- base::NROW(energie)
      ylim <- range(energie)
      
      #partition the graphic window in order to add the legend on the right
      graphics::layout(mat = base::matrix(1:2,1,2), widths = c(0.6,0.4))
      graphics::par(mar = c(5.1,4.1,2.1,1.1), cex = scale, lwd = scale)
      
      #empty plot
      base::plot(NULL,xlim = c(1,xlim), ylim = ylim, xlab = "Iteration", ylab = "Energy", main = "",bty = "L")
      
      #add each energy chain plot
      for(i in seq_along(which_chains)){
        graphics::lines(1:xlim,energie[,i], col = colori[which_chains[i]])
      }
      
      #add the legend on the right
      graphics::par(mar = c(1.1,1.1,1.1,1.1), cex = scale, lwd = scale)
      base::plot(1, type = "n", axes = FALSE, ylab = "", xlab = "", bty = "n")
      graphics::legend("top",bty = "n", bg = "transparent", title = "Chain",
                       legend = which_chains, col = colori[which_chains], lty = 1,
                       lwd = 2, title.cex = 1, cex = 1)
      
    }else if(type == 4){
      #plot4: stick plot of trajectory length
      
      #gets trajectory length frequency for each chain
      vals <- base::lapply(which_chains,function(idx) {
        tmp <- base::table(x$chains[[idx]][["step_length"]])
        base::cbind(as.numeric(names(tmp)),tmp)
      })
      
      #compute x and y limits
      x_range <- range(c(1,base::sapply(vals,function(x) max(x[,1]))))
      y_range <- c(0,max(base::sapply(vals,function(x) max(x[,2]))))
      
      #empty plot
      base::plot(NULL,xlim = x_range, ylim = y_range,  xlab = "L",
                 ylab = "Frequency",main = "Iteration's Step-Length", bty = "L",
                 cex.main = 1)
      
      #add each chain stick plot
      for(i in seq_along(vals)){
        for(j in seq_len(NROW(vals[[i]]))){
          graphics::lines(rep(vals[[i]][j,1],2)+0.1*(i-1) , c(0,vals[[i]][j,2]), col = colori[which_chains[i]], lwd = 3)
        }
      }
      
      #add legend
      graphics::legend("topright",bty = "n", bg = "transparent", title = "Chain",
                       legend = (1:nc)[which_chains], col = colori[which_chains], lty = 1, lwd = 2, cex = 1)
      
    }else if(type == 5){
      
      #plot5: marginal and first difference energy histogram
      
      #get energy chains
      E <- base::sapply(which_chains,function(i) x$chains[[i]]$energy)
      
      #get first difference energy chains
      delta_E <- base::sapply(which_chains,function(i) x$chains[[i]]$delta_energy)
      
      #get sample size
      N <- prod(dim(E))
      
      #compute the number of bins
      n <- min(100,N / 33)
      
      #compute marginal energy frequencies
      g1 <- graphics::hist(c(E),n = n, plot = FALSE)
      
      #center them on their mode
      g1$mids <- g1$mids - g1$mids[base::which.max(g1$density)[1]]
      
      #compute first difference energy frequencies
      g2 <- graphics::hist(c(delta_E),n = n, plot = FALSE)
      
      #center them on their mode
      g2$mids <- g2$mids - g2$mids[base::which.max(g2$density)[1]]
      
      #compute x and y limits
      xlim <- range(g1$mids,g2$mids)
      ylim <- c(0,max(g1$density,g2$density))
      
      #compute an adeguate span between each stick
      span <- min(0.5,base::diff(xlim)/n/2)
      
      #empty plot
      base::plot(NULL,xlim = xlim,ylim = ylim, xlab = "Centered Energy", ylab = "Frequency", bty = "L")
      
      #add marginal histogram
      for(ii in seq_along(g1$breaks)){
        graphics::lines(rep(g1$mids[ii],2),c(0,g1$density[ii]), lwd = 2, col = grDevices::adjustcolor("darkgray",0.7))
      }
      
      #add first differences histogram
      for(ii in seq_along(g2$breaks)){
        graphics::lines(rep(g2$mids[ii],2) + span,c(0,g2$density[ii]), lwd = 2, col = grDevices::adjustcolor("darkred",0.7))
      }
      
      #add legend
      Delta <- NULL #in order to avoid stupid CRAN problems
      graphics::legend("topright",bty = "n", bg = "transparent", title = "",
                       legend = c("E",expression(paste(Delta,"E"))), 
                       col = grDevices::adjustcolor(c("darkgray","darkred",0.7)), lty = 1, lwd = 2, cex = 1)
    }else if(type == 6){
      #plot6: autocorrelation plots
      
      #get lag maximum number 
      lag_max <- floor(10 * log10(base::NROW(x$chains[[1]]$values)))
      
      #compute autocorrelation for each chain and parameter specified
      acfs <- base::lapply(which,function(i) 
        base::sapply(x$chains[which_chains], function(XX){
          tmp <- stats::acf(XX[[quale]][,i], plot = FALSE,lag.max = lag_max)$acf[-1,,1]
          #set NaN values to NA
          tmp[is.nan(tmp)] <- NA
          tmp
        }))
      
      #if there are NAs report it
      if(any(base::sapply(acfs,function(x) any(is.na(x))))){
        base::message("NA values, probably due to some reducible chain.\nThe corresponding ACFs are not well-defined so these are not plotted.")
      }
      
      #compute for each parameter the y limits
      ylim <- base::sapply(acfs,range,na.rm = TRUE)
      
      #let's partition the graphics window accordingly
      k1 <- ceiling(sqrt(d))
      if(d <= k1*(k1-1)) {
        k2 <- k1 - 1
      }else{
        k2 <- k1
      }
      graphics::layout(mat = rbind(1,cbind(matrix(1 + 1:(k1*k2),k2,k1, byrow = TRUE),k1*k2+2)),
                       heights = c(max(0.25,1-1/k2),rep(1,k2)))
      
      #add title
      graphics::par(mar = c(1.1,1.1,1.1,1.1), cex = scale, lwd = scale)
      graphics::plot.new()
      graphics::text(0.5,0.5,"Autocorrelation Plots",cex=1,font=2)
      
      graphics::par(mar = c(1.1,2.1,3.1,1.1), cex = scale, lwd = scale)
      
      #add each parameter plot
      for(i in seq_along(which)){
        #empty plot
        base::plot(NULL,xlim = c(1,lag_max),
                   ylim = range(0,ylim[,i],1), main = base::colnames(x$chains[[1]]$values)[which[i]],
                   xlab = "", ylab = "")
        #autocorrelation of each chain
        for(j in seq_along(which_chains))
          graphics::lines(1:lag_max, acfs[[i]][,j], col = colori[which_chains[j]])
      }
      
      #remaining empty plot
      for(i in seq_len(k2*k1 - d)){
        graphics::plot.new()
      }
      
      #add legend
      graphics::par(mar = c(1.1,1.1,1.1,1.1), cex = scale, lwd = scale)
      base::plot(1, type = "n", axes = FALSE, ylab = "", xlab = "", bty = "n")
      graphics::legend("top", title = "Chain", legend = which_chains, col = colori[which_chains],
                       bty = "n", lty = 1, lwd = 2, cex = 1, title.cex = 1,bg = "transparent")
      
    }else if(type == 7){
      #plot7: matplot of alphas
      
      #check the number of chains
      if(length(which_chains) == 1){
        stop("alphas plots has no sense with only one chain!")
      }
      
      #consider all the possible cases
      if(x$k == 0 || (x$d > x$k && all(which == 1))){
        #only the global empirical rate
        
        titolo <- "Empirical Global Acceptance Rate"
        nomi <- "0"
      }else if(x$k == x$d || (x$d > x$k && all(which != 1))){
        #only the refraction rates
        if(length(which) == 1){
          #only one
          
          titolo <- "Empirical Refraction Rate"
          nomi <- base::as.character(which)
        }else{
          #many
          
          titolo <- "Empirical Refraction Rates" 
          nomi <- base::as.character(which)
        }
      }else{
        #both the refraction and global rates
        
        titolo <- "Empirical Rates"
        nomi <- base::as.character(which-1)
      }
      
      graphics::par(mar = c(4.1,4.1,5.1,2.1), xpd = TRUE, cex = scale, lwd = scale)
      #discriminate different cases
      if(x$k == 0 || length(which) == 1 ){
        #no legend
        graphics::matplot(base::t(base::do.call(base::cbind,base::sapply(x$chains[which_chains],"[","alpha"))), 
                          type = "l", col = colori, lty = 1,
                          ylim = c(0,1), xlab = "Chain", ylab = "Empirical rate",
                          bty = "L", main = titolo, cex = 1)
        
      }else if(x$k > 0 && x$k < x$d && any(which == 1)){
        #use different linetype
        graphics::matplot(base::t(base::do.call(base::cbind,base::sapply(x$chains[which_chains],"[","alpha"))), 
                          type = "l", col = colori, lty = c(1,rep(2,base::length(which)-1)),
                          ylim = c(0,1), xlab = "Chain", ylab = "Empirical rate",
                          bty = "L", main = titolo, cex = 1)
        
        #legends
        graphics::legend("topleft", title = "Type", legend = c("global","refraction"),
                         col = 1, lty = 1:2, lwd = 2, bty = "n", bg = "transparent", inset = c(0,-0.2))
        graphics::legend("topright", title = "", legend = nomi,
                         col = colori, lty = 1:2, lwd = 2, bty = "n", bg = "transparent", inset = c(0,-0.2))
        
      }else{
        #use the same linetype
        graphics::matplot(base::t(base::do.call(base::cbind,base::sapply(x$chains[which_chains],"[","alpha"))), 
                          type = "l", col = colori, lty = 1,
                          ylim = c(0,1), xlab = "Chain", ylab = "Empirical rate",
                          bty = "L", main = titolo, cex = 1)
        
        #legends
        graphics::legend("topright", title = "", legend = nomi,
                         col = colori, lty = 1:2, lwd = 2, bty = "n", bg = "transparent", inset = c(0,-0.2))
        
      }
      
    }else{
      base::message("No such plot, 'type' argument accept value between 1 and 7!")
      return(NULL)
    }
  }
  
}

### SUMMARY FUNCTION OF THE MCMC OUTPUT 
#' Function to print the summary of an XDNUTS model.
#'
#' @param object an object of class \code{XDNUTS}.
#' @param ... additional arguments to customize the summary.
#' @param q.val desired quantiles of the posterior distribution for each coordinate.
#' Default values are \code{0.05,0.25,0.5,0.75,0.95}.
#'
#'@param which either a numerical vector indicating the index of the parameters of interest or a string \describe{
#' \item{\code{which = 'continuous'}}{for plotting the first \eqn{d-k} parameters.}
#' \item{\code{which = 'discontinuous'}}{for plotting the last \eqn{k} parameters.}
#' }
#' where both \eqn{d} and \eqn{k} are elements contained in the output of the function \link{xdnuts}.
#' @param which_chains a numerical vector indicating the index of the chains of interest.
#'
#' @return a list containing a data frame named \code{stats} with the following columns: \describe{\item{mean}{the mean of the posterior distribution.}
#' \item{sd}{the standard deviation of the posterior distribution.}
#' \item{q.val}{the desired quantiles of the posterior distribution.}
#' \item{ESS}{the Effective Sample Size for each marginal distribution.}
#' \item{R_hat}{the Potential Scale Reduction Factor of Gelman \insertCite{gelman1992inference}{XDNUTS}, only if multiple chains are available.}
#' \item{R_hat_upper_CI}{the upper confidence interval for the latter, only if multiple chains are available.}
#' }
#' Other quantities returned are:\describe{
#' \item{Gelman.Test}{the value of the multivariate Potential Scale Reduction Factor test \insertCite{gelman1992inference}{XDNUTS}.}
#' \item{BFMI}{the value of the empirical Bayesian Fraction of Information Criteria \insertCite{betancourt2016diagnosing}{XDNUTS}. 
#' A value below 0.2 indicates a bad random walk behavior in the energy Markov Chain, mostly due to a suboptimal
#' specification of the momentum parameters probability density.}
#' \item{n_div}{the total number of trajectory ended with a divergent transition.}
#' \item{n_premature}{the total number of trajectory with a premature termination.}}
#'
#' @references 
#'  \insertAllCited{} 
#' 
#' @export summary.XDNUTS
#' @export
summary.XDNUTS <- function(object, ..., q.val = c(0.05,0.25,0.5,0.75,0.95),
                           which = NULL, which_chains = NULL){
  
  #check inputs
  if(!base::inherits(object,"XDNUTS")){
    base::stop("'object' must be of class 'XDNUTS'!")
  }
  
  if(!is.numeric(q.val) || any(q.val > 1 | q.val < 0) || length(q.val) < 1){
    base::stop("'q.val' must be a numeric vector with values bounded in [0,1]!")
  }

  #get the initial number of chains
  nc <- length(object$chains)
  
  #get the indexes of desired chains and make sure they are admissible
  if(is.null(which_chains)){
    which_chains <- seq_len(nc)
  }else{
    if(any(which_chains > nc | which_chains < 1)){
      base::stop("Incorrect chain indexes!")
    }
    which_chains <- base::unique(which_chains)
  }
  
  #which parameters do we want to see the summary of?
  #make sure the input in admissible
  if(is.null(which)){
    which <- seq_len(base::NCOL(object$chains[[1]]$values))
  }else if(all(which == "continuous")){
    
    which <- seq_len(object$d)[seq_len(object$d-object$k)]
    
  }else if(all(which == "discontinuous")){
    
    which <- seq_len(object$d)[-seq_len(object$d-object$k)]
    
  }else if(any(which < 1 | which > base::NCOL(object$chains[[1]]$values)) ){
    base::stop("Incorrect index of parameters!")
  }
  
  #transformation of the origian output in a mcmc.list object of coda package
  res <- list()
  conta <- 1
  for(i in which_chains){
    res[[conta]] <- coda::mcmc(object$chains[[i]]$values[,which,drop = FALSE])
    conta <- conta + 1
  }
  res <- coda::as.mcmc.list(res)
  
  #compute petential scale reduction factor
  gelman.test <- base::tryCatch(coda::gelman.diag(res), error = function(x) list(psrf = NULL, mpsrf = NULL))
  
  #compute posterior statistics
  out <- base::t(base::apply(base::do.call(base::rbind,res),2,function(x) 
    c(base::mean(x),stats::sd(x),stats::quantile(x,q.val))))
  out <- base::cbind(out,coda::effectiveSize(res),gelman.test$psrf)
  out <- base::as.data.frame(out)
  
  #if there are more then 1 chain, add the Rhat statistics
  if(conta > 2 && !is.null(gelman.test$psrf)){
    base::colnames(out) <- c("mean","sd",base::paste0(q.val*100,"%"),"ESS","R_hat","R_hat_upper_CI")
  }else{
    base::colnames(out) <- c("mean","sd",base::paste0(q.val*100,"%"),"ESS")
  }
  
  #compute the empirical bayesian fraction of missing information:
  
  #get energy
  E <- base::sapply(object$chains,function(x) x$energy)
  
  #get first difference energy
  delta_E <- base::sapply(object$chains,function(x) x$delta_energy)
  
  #estimate it
  BFMI <- stats::var(c(E)) / stats::var(c(delta_E))
  
  #build output list
  out <- list(stats = out, Gelman.Test = gelman.test$mpsrf,BFMI = BFMI)
  
  #count the number of divergent transitions
  out$n_divergence <- sum(base::sapply(object$chains,function(x) sum(base::NROW(x$div_trans))))

  #compute the number of trajectories terminated prematurely
  out$n_premature <- sum( c(base::sapply(object$chains,function(x) 
    x$step_length)) == (2^object$control$max_treedepth - 1))

  #make the list of class summary.XDNUTS
  class(out) <- "summary.XDNUTS"
  
  #return the list
  return(out)
  
}

### FUNCTION FOR PRINT OF AN summary.XDNUTS OBJECT
#' Function for printing an object of class summary.XDNUTS
#' 
#' Print to console the statistics about the MCMC samples obtained with
#' a call to the function \link{summary.XDNUTS}. See that for details.
#' 
#' @param x an object of class summary.XDNUTS
#' @param ... additional values to pass. These currently do nothing.
#' @param digits number of digits for rounding the output. Default value is 3.
#' 
#' @return No return value.
#' 
#' @export print.summary.XDNUTS
print.summary.XDNUTS <- function(x,... , digits = 3){
  
  #check input
  if(!base::inherits(x,"summary.XDNUTS")){
    base::stop("'x' must be an object of class 'summary.XDNUTS!'")
  }
  
  if(!is.numeric(digits) || length(digits) > 1){
    base::stop("'digits' must be a scalar integer value!")
  }
  
  #print to console the table
  base::print(round(x$stats,digits = digits))
  if(!is.null(x$Gelman.Test)){
    #if available, add the multivariate test statistic
    base::cat("\nMultivariate Gelman Test: ",round(x$Gelman.Test, digits = digits))
  }
  
  #print BFMI to console
  base::cat("\nEstimated Bayesian Fraction of Missing Information: ",round(x$BFMI, digits = digits),"\n")
  
  #print warnings message
  if(x$n_divergence > 0){
    #divergent transitions
    base::message("\n",x$n_divergence, " trajectory ended with a divergent transition!\nConsider increasing 'control$delta' via 'set_parameters' to reduce bias.")
  }
  if(x$n_premature > 0){
    #premature ending trajectory
    base::message("\n",x$n_premature, " trajectory ended before reaching an effective termination!\nFlat regions, consider increasing 'control$max_treedepth' via 'set_parameters'.")
  }
}


### FUNCTION FOR CONVENIENT SAMPLE EXTRACTION
#' Function to extract samples from the output of an XDNUTS model.
#'
#' @param X an object of class \code{XDNUTS}.
#'
#'@param which either a numerical vector indicating the index of the parameters of interest or a string \describe{
#' \item{\code{which = 'continuous'}}{for plotting the first \eqn{d-k} parameters.}
#' \item{\code{which = 'discontinuous'}}{for plotting the last \eqn{k} parameters.}
#' }
#' where both \eqn{d} and \eqn{k} are elements contained in the output of the function \link{xdnuts}.
#'
#' @param which_chains a vector of indices containing the chains to extract. By default, all chains are considered.
#'
#' @param collapse a boolean value. If TRUE, all samples from every chain are collapsed into one. The default value is FALSE.
#' @param thin an integer value indicating how many samples should be discarded before returning an iteration of the chain.
#' @param burn an integer value indicating how many initial samples for each chain to burn.
#' 
#' @return an \eqn{N \times d} matrix or an \eqn{N \times d \times C} array, where C is the number of chains, containing the MCMC samples.
#'
#' @export xdextract
xdextract <- function(X, which = NULL, which_chains = NULL,
                      collapse = FALSE, thin = NULL, burn = NULL){
  
  #check inputs
  if(!base::inherits(X,"XDNUTS")){
    base::stop("'X' must be an object of class 'XDNUTS'!")
  }
  
  if(!is.logical(collapse) || length(collapse) > 1){
    base::stop("'collapse' must be a logical scalar value!")
  }
  
  #get the number of chains
  nc <- length(X$chains)
  
  #get the length of one chain
  chain_length <- base::NROW(X$chains[[1]]$values)
  
  #process the burn argument
  if(!is.null(burn)){
    
    if(length(burn) > 1 || !is.numeric(burn) || any(burn < 0)){
      base::stop("'burn' must be a positive integer scalar value!")
    }
    
    #make sure that this value isn't greater than sample size
    if(burn > chain_length){
      base::stop("'burn' can't be greater than the Monte Carlo sample size!")
    }
    
  }else{
    
    #set the default value to zero
    burn <- 0
  }
  
  #process the thin argument
  if(!is.null(thin)){
    
    if(length(thin) > 1 || !is.numeric(thin) || any(thin < 0)){
      base::stop("'thin' must be a positive integer scalar value!")
    }
    
    #make sure that this value isn't greater than sample size
    if(thin > chain_length - burn){
      base::stop("'thin' can't be greater than the Monte Carlo sample size!")
    }else{
      idx_thin <- base::seq(burn+1,chain_length,by = thin)
    }
  }else{
    
    #get all samples
    idx_thin <- (burn+1):chain_length
  }
  
  #get chain indexes to extract and make sure they are admissible
  if(is.null(which_chains)){
    which_chains <- seq_len(nc)
  }else{
    if(any(which_chains > nc | which_chains < 1)){
      base::stop("Incorrect chain indexes!")
    }
    which_chains <- base::unique(which_chains)
  }
  
  #of which parameters do we want to extract samples?
  #make sure they are admissible
  if(is.null(which)){
    which <- seq_len(base::NCOL(X$chains[[1]]$values))
  }else if(all(which == "continuous")){
    
    which <- seq_len(X$d)[seq_len(X$d-X$k)]
    
  }else if(all(which == "discontinuous")){
    
    which <- seq_len(X$d)[-seq_len(X$d-X$k)]
    
  }else if(any(which < 1 | which > base::NCOL(X$chains[[1]]$values)) ){
    base::stop("Incorrect index of parameters!")
  }
  
  if(!collapse){
    #return an array
    
    #initialize the array
    out <- base::array(NA,dim = c(length(idx_thin),
                            length(which),
                            length(which_chains)))
    
    #fill it
    conta <- 1
    for(i in which_chains){
      out[,,conta] <- X$chains[[i]]$values[idx_thin,which, drop = FALSE]
      conta <- conta + 1
    }
    
    #assign proper dimension names
    dimnames(out) <- list(NULL,base::colnames(X$chains[[1]]$values)[which],
                          base::paste0("chain_",which_chains))
  }else{
    #return a matrix
    
    #initialize the matrix
    out <- base::matrix(NA,length(idx_thin)*length(which_chains),length(which))
    
    #fill it
    conta <- 0
    for(i in which_chains){
      out[conta*length(idx_thin) + 1:length(idx_thin),] <- X$chains[[i]]$values[idx_thin,which, drop = FALSE]
      conta <- conta + 1
    }
    
    #assign proper dimension names
    base::colnames(out) <- base::colnames(X$chains[[1]]$values)[which]
  }
  
  #return the array/matrix
  return(out)
}

### FUNCTION THAT APPLIES A TRANSFORMATION TO THE CHAINS
#' Function to apply a transformation to the samples from the output of an XDNUTS model.
#'
#' @param X an object of class \code{XDNUTS}.
#' @param which a vector of indices indicating which parameter the transformation should be applied to.
#'  If \code{NULL}, the function is applied to all of them.
#' @param FUN a function object which takes one or more components of an MCMC iteration and any other possible arguments.
#' @param ... optional arguments for FUN.
#' @param new.names a character vector containing the parameter names in the new parameterization.  
#'  If only one value is provided, but the transformation involves more, the name is iterated with an increasing index.
#' @param thin an integer value indicating how many samples should be discarded before returning an iteration of the chain.
#' @param burn an integer value indicating how many initial samples for each chain to discard.
#' 
#' @return an object of class \code{XDNUTS} with the specified transformation applied to each chain.
#'
#' @export xdtransform
xdtransform <- function(X, which = NULL, FUN = NULL, ...,
                        new.names = NULL, thin = NULL, burn = NULL){
  
  #check inputs
  if(!base::inherits(X,"XDNUTS")){
    base::stop("'X' must be an object of class 'XDNUTS'!")
  }
  
  if(is.null(FUN)){
    FUN <- function(x) x
  }
  
  if(!is.function(FUN)){
    base::stop("'FUN' must be a function object!")
  }
  
  #copy the original XDNUTS object
  out <- X
  
  #get old parameters names
  old_names <- colnames(X$chains[[1]]$values)
  
  #get the length of one chain
  chain_length <- base::NROW(X$chains[[1]]$values)
  
  #process the burn argument
  if(!is.null(burn)){
    
    if(length(burn) > 1 || !is.numeric(burn) || any(burn < 0)){
      base::stop("'burn' must be a positive integer scalar value!")
    }
    
    #make sure that this value isn't greater than sample size
    if(burn > chain_length){
      base::stop("'burn' can't be greater than the Monte Carlo sample size!")
    }
    
  }else{
    
    #set the default value to zero
    burn <- 0
  }
  
  #process the thin argument
    if(!is.null(thin)){
    
      if(length(thin) > 1 || !is.numeric(thin) || any(thin < 0)){
        base::stop("'thin' must be a positive integer scalar value!")
      }  
      
      #make sure that this value isn't greater than sample size
      if(thin > chain_length - burn){
        base::stop("'thin' can't be greater than the Monte Carlo sample size!")
      }else{
        idx_thin <- base::seq(burn+1,chain_length,by = thin)
    }
  }else{
    
    #get all samples
    idx_thin <- (burn+1):chain_length
  }
  
  #update the number of iteration in the new XDNUTS object and the thin attribute
  out$N <- length(idx_thin)
  out$thin <- thin
  
  #ensure argument which is admissible
  if(!is.null(which)){
    
    if(is.character(which)){
      #character vector case
      if(any(!sapply(which,function(x) x %in% old_names))){
        base::stop("Incorrect parameter names specified!")
      }else{
        which <- base::sapply(which,function(x) base::which(old_names == x))
      }
    }else if(is.numeric(which)){
      #numeric vector case
      if(any(which < 1 | which > base::NCOL(X$chains[[1]]$values))){
        base::stop("Incorrect index of parameters")
      }
    }else{
      base::stop("Wrong 'which' type!")
    }
  }else{
    which <- seq_along(old_names)
  }
  
  #case when we update all the components
  if(length(which) == length(old_names) && !is.null(old_names)){
    
    #loop for every chain and apply the transformation
    for(cc in seq_along(X$chains)){
      
      #apply transformation
      tmp <- base::apply(X$chains[[cc]]$values[idx_thin,] , 1 , FUN , ... )
      
      #check if the object is a matrix
      if(is.matrix(tmp)){
        #check if the dimension is one
        if(dim(tmp)[1] > 1){
          #if so, adjust with the transpose
          tmp <- base::t(tmp)
        }
      }else{
        #make it a matrix
        tmp <- as.matrix(tmp)
      }
      
      out$chains[[cc]]$values <- tmp
      #give names to the new parameters
      if(is.null(new.names)){
        base::colnames(out$chains[[cc]]$values) <- 
          base::paste0("theta",seq_len(base::NCOL(out$chains[[cc]]$values)))
      }else{
        if(length(new.names) == 1 && base::NCOL(out$chains[[cc]]$values) != 1){
          base::colnames(out$chains[[cc]]$values) <- 
            base::paste0(new.names,seq_len(base::NCOL(out$chains[[cc]]$values)))
        }else if(length(new.names) == base::NCOL(out$chains[[cc]]$values) ){
          base::colnames(out$chains[[cc]]$values) <- new.names
        }else{
          base::stop("Incorrect number of values in 'new.names'!")
        }
      }
      
      #do the same, eventually, for the warm up phase
      if(!is.null(out$chains[[1]]$warm_up)){
        
        #get the length of the warm up chain
        chain_length <- base::NROW(out$chains[[1]]$warm_up)
          
        #if the thinning is greater don't apply it
        if(thin > chain_length){
          idx_thin <- seq_len(chain_length)
        }else{
          idx_thin <- base::seq(1,chain_length,by = thin)
        }
        
        #apply transformation
        tmp <- base::apply(X$chains[[cc]]$warm_up[idx_thin,] , 1 , FUN , ... )
        
        #check if the object is a matrix
        if(is.matrix(tmp)){
          #check if the dimension is one
          if(dim(tmp)[1] > 1){
            #if so, adjust with the transpose
            tmp <- base::t(tmp)
          }
        }else{
          #make it a matrix
          tmp <- as.matrix(tmp)
        }
        
        out$chains[[cc]]$warm_up <- tmp
        
        #give names to the new parameters
        if(is.null(new.names)){
          base::colnames(out$chains[[cc]]$warm_up) <- 
            base::paste0("theta",seq_len(base::NCOL(out$chains[[cc]]$warm_up)))
        }else{
          if(length(new.names) == 1 && base::NCOL(out$chains[[cc]]$warm_up) != 1){
            base::colnames(out$chains[[cc]]$warm_up) <- 
              base::paste0(new.names,seq_len(base::NCOL(out$chains[[cc]]$warm_up)))
          }else if(length(new.names) == base::NCOL(out$chains[[cc]]$warm_up) ){
            base::colnames(out$chains[[cc]]$warm_up) <- new.names
          }else{
            base::stop("Incorrect number of values in 'new.names'!")
          }
        }
      }
    }
    
  }else{
    #case when we update only some components
    
    if(!is.null(out$chains[[1]]$warm_up)){
      
      #get the length of the warm up chain
      chain_length_warm <- base::NROW(out$chains[[1]]$warm_up)
      
      #if the thinning is greater don't apply it
      if(!is.null(thin)){
        if(thin > chain_length_warm){
          idx_thin_warm <- seq_len(chain_length_warm)
        }else{
          idx_thin_warm <- base::seq(1,chain_length_warm,by = thin)
        }
      }else{
        idx_thin_warm <- seq_len(chain_length_warm)
      }
    }
    
    #loop for every chain and apply the transformation
    for(cc in seq_along(X$chains)){
      out$chains[[cc]]$values[,which] <- 
        (base::apply(X$chains[[cc]]$values[idx_thin,which,drop = FALSE] , 2 , FUN , ... ))
    
      #give names to the new parameters
      if(is.null(new.names)){
        base::colnames(out$chains[[cc]]$values)[which] <- 
          base::paste0("theta",seq_along(which))
      }else{
        if(length(new.names) == 1 && length(which) != 1){
          base::colnames(out$chains[[cc]]$values)[which] <- 
            base::paste0(new.names,seq_along(which))
        }else if(length(new.names) == length(which) ){
          base::colnames(out$chains[[cc]]$values)[which] <- new.names
        }else{
          base::stop("Incorrect number of values in 'new.names'!")
        }
      }
      
      #do the same, eventually, for the warm up phase
      if(!is.null(out$chains[[cc]]$warm_up)){
        
        out$chains[[cc]]$warm_up[,which] <- 
          (base::apply(X$chains[[cc]]$warm_up[idx_thin_warm,which,drop = FALSE] , 2 , FUN , ... ))
        
        #give names to the new parameters
        if(is.null(new.names)){
          base::colnames(out$chains[[cc]]$warm_up)[which] <- 
            base::paste0("theta",seq_along(which))
        }else{
          if(length(new.names) == 1 && length(which) != 1){
            base::colnames(out$chains[[cc]]$warm_up)[which] <- 
              base::paste0(new.names,seq_along(which))
          }else if(length(new.names) == length(which) ){
            base::colnames(out$chains[[cc]]$warm_up)[which] <- new.names
          }else{
            base::stop("Incorrect number of values in 'new.names'!")
          }
        }
      }
    }
  }
  

  #return the new XDNUTS object
  return(out)
}
