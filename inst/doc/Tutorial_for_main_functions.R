## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = NULL
)

## ----eval = FALSE-------------------------------------------------------------
# theta0 <- lapply(seq_len(N_chains),myfunction)

## ----echo = FALSE, eval = FALSE-----------------------------------------------
# set.seed(123)

## ----setup, eval = TRUE, echo = FALSE-----------------------------------------
library(XDNUTS)

## ----eval = TRUE--------------------------------------------------------------
#observed data
X <- 50

#hyperparameteers
a0 <- 10
b0 <- 10

#list of arguments
arglist <- list(data = X, hyp =  c(a0,b0))

## ----eval = TRUE--------------------------------------------------------------
#function for the negative log posterior and its gradient
#with respect to the continuous components
nlp <- function(par,args,eval_nlp = TRUE){
  
  if(eval_nlp){
    #only the negative log posterior
    
    #overflow
    if(any(par > 30)) return(Inf) 
    
    #conversion of the r value
    r <- ceiling(1 + args$data*plogis(par[2])) - 1
    
    #ensure that r is not zero
    if(r == 0){
      r <- 1
    }
    
    #output
    out <- sum(log(seq_len(r-1))) + 
      (args$data + args$hyp[1] + args$hyp[2])*log(1+exp(-par[1])) + 
      par[1]*(args$data - r + args$hyp[2]) + par[2] + 2*log(1+exp(-par[2]))
    if(r > 2) out <- out - sum(log(seq(args$data - r + 1,args$data - 1)))
    
    return(out)
    
  }else{
    #only the gradient
    
    #overflow
    if(any(par > 30)) return(Inf) 
    
    #conversion of the r value
    r <- ceiling(1 + args$data*plogis(par[2])) - 1
    
    #conversion of the r value
    r <- ceiling(1 + args$data*plogis(par[2])) - 1
    
    #output
    return( (args$data - r + args$hyp[2]) - 
              (args$data + args$hyp[1] + args$hyp[2])*(1-plogis(par[1])) )
  }
  
}

## ----warning=FALSE, message=FALSE, eval = TRUE--------------------------------
#MCMC
set.seed(1)
chains <- xdnuts(theta0 = lapply(1:4,function(x) c(omega = rnorm(1),r_hat = rnorm(1))),
                 nlp = nlp,
                 args = arglist,
                 k = 1,
                 thin = 1,
                 parallel = FALSE,
                 method = "NUTS",
                 hide = TRUE)

## ----eval = TRUE--------------------------------------------------------------
chains

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains, type = 2, gg = FALSE)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains, type = 3)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains, type = 4)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains, type = 5)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains, type = 6)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains, type = 7)

## ----warning=FALSE, message=FALSE, out.width='70%',out.height='70%', eval = TRUE----
summary(chains)

## ----eval = TRUE, dpi = 300---------------------------------------------------
#MCMC
set.seed(1)
chains <- xdnuts(theta0 = lapply(1:4,function(x) rnorm(2)),
                 nlp = nlp,
                 args = arglist,
                 k = 2,
                 thin = 1,
                 parallel = FALSE,
                 method = "XHMC",
                 hide = TRUE,
                 tau = 0.01)

## ----eval = TRUE--------------------------------------------------------------
#define the function to be applied to each sample
f <- function(x,XX) {
  c(
    plogis(x[1]), #inverse logistic for the probability
    ceiling(1 + XX*plogis(x[2])) - 1 #number of trials
  )
}
original_chains <- xdtransform(X = chains, which = NULL,
                               FUN = f,XX = arglist$data,
                               new.names = c("p","r"))

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 2, gg = FALSE)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 4)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 6)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
summary(original_chains)

## ----eval = TRUE--------------------------------------------------------------
data(viscosity)
viscosity

#create the list function
arglist <- list(data =  as.matrix(viscosity[,-1]),
                hyp = c(0, #mean a priori for mu
                        1000, #variance a priori for mu
                        0.5,1,0.5,1 #inverse gamma iperparameters
                        )
                )

## ----eval = TRUE--------------------------------------------------------------
nlp <- function(par,args,eval_nlp = TRUE){
    
    if(eval_nlp){
      #only the negative log posterior

      #overflow
      if(any(abs(par[2:3]) > 30)) return(Inf)
      
      return(par[2] * ( prod(dim(args$data)) + args$hyp[3] ) + 
               (sum( (args$data - par[-(1:3)])^2 ) + 
                  2*args$hyp[4])*exp(-par[2])/2 +
               par[3] * (length(par[-(1:3)]) + 
                           args$hyp[5]) +  
               (sum( (par[-(1:3)] - par[1])^2 ) + 
                  2+args$hyp[6])*exp(-par[3])/2 +
               (par[1] - args$hyp[1])^2/2/args$hyp[2])
      
    }else{
      #only the gradient
      
      #overflow
      if(any(abs(par[2:3]) > 30)) return(rep(Inf,9))
      
      c(
        -sum( par[-(1:3)] - par[1] )*exp(-par[3]) + (
          par[1] - args$hyp[1])/args$hyp[2], #mu derivative
        
        prod(dim(args$data)) + args$hyp[3] - 
          (sum( (args$data - par[-(1:3)])^2 ) +
             2*args$hyp[4])*exp(-par[2])/2, #omega derivative
        
        length(par[-(1:3)]) + args$hyp[5] - 
          (sum( (par[-(1:3)] - par[1])^2 ) + 
             2+args$hyp[6])*exp(-par[3])/2, #omega_a derivative
        
        -apply(args$data - par[-(1:3)],1,sum)*exp(-par[2]) + 
          (par[-(1:3)] - par[1])*exp(-par[3]) #random effects gradient
      )
      
    }
  
}

## ----eval = TRUE--------------------------------------------------------------
#MCMC
set.seed(1)
chains <- xdnuts(theta0 = lapply(1:4,function(x) {
                      out <- rnorm(3 + 6)
                      names(out) <- c("mu","log_s2","log_s2_a",
                                      paste0("mu",1:6))
                      out}),
                 nlp = nlp,
                 args = arglist,
                 k = 0, #no discontinuous components
                 thin = 1,
                 parallel = FALSE,
                 method = "HMC",
                 hide = TRUE,
                 L = 31,
                 control = set_parameters(L_jitter = 5,
                                          N_adapt = 1e3,
                                          keep_warm_up = TRUE))

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(chains,warm_up = TRUE)

## ----eval = TRUE--------------------------------------------------------------
original_chains <- xdtransform(X = chains,which = 2:3,
                               FUN = exp,new.names = c("s2","s2_a"))

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 3)

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 2, which = 1:3, gg = FALSE, scale = 0.5)#fixed

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 2, which = 4:9, gg = FALSE, scale = 0.5)#random

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
plot(original_chains, type = 6)

## ----out.width='70%', out.height='70%', eval = TRUE---------------------------
summary(original_chains, q.val = c(0.05,0.5,0.95))

## ----out.width='70%', out.height='70%', eval = TRUE, dpi = 300----------------
#extract samples into matrix
theta <- xdextract(original_chains,collapse = TRUE)

#compute prediction
y_hat <- sapply(1:6, function(i){
  rnorm(NROW(theta),theta[,3+i],sqrt(theta[,2]))
})

#plot prediction
op <- par(no.readonly = TRUE)
par(mar=c(5.1, 4.1, 2.1, 4.1), xpd=TRUE)
plot(NULL, xlim = c(1,6), ylim = c(15,85), xlab = "Subject",
     ylab =  "Viscosity", bty = "L")
for(i in 1:6){
  #data
  points(rep(i,7),viscosity[i,-1], pch = 16)
  
  #data 95% credible intervals for the prediction
  lines(rep(i,2),quantile(y_hat[,i],c(0.025,0.975)), col = 3, lwd = 3)
  
  #random effects 95% credible intervals
  lines(rep(i,2),quantile(theta[,3+i],c(0.025,0.975)), col = 4, lwd = 4)
}
legend("topright",inset=c(-0.2,-0.2), lty = 1, lwd = 2, col = c(3,4),
       title = "95% Credible Intervals",
       legend = c("prediction","random effects"),
       bty = "n", bg = "transparent", cex = 0.8)
par(op)

