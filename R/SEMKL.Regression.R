#' Prediction from MKL model
#'
#' This makes prediction for multiple kernel regression
#' 
#' @param k Gramm matrix of training data
#' @param outcome observed dependent variable
#' @param penalty Cost of unit miss fitted - observed
#' @param epsilon SVM parameter defining support vectors
#' @param tol Convergence criteria, algorithm stops once the biggest change of kernel weights
#' in two consectutive iterations is less than tol.
#' @param max.iters Termination criteria, algorithm will stop after 1000 iterations
#' @return results Returns a list which includes model parameters, weights for kernels, and 
#' f which are the fitted values for the training set
#' @export
#' @examples 
#' \dontrun{
#' library(kernlab)
#' x=as.matrix(10*runif(200),ncol=1)
#' y=x*sin(x)+rnorm(200)
#' plot(x,y)
#'
#" K=kernels.gen(data=x, train.samples = 1:150, kernels = c('radial','radial','linear'), 
#"              sigma = c(2,1/25,0))
#'
#' model=SEMKL.regression(k=K$K.train, outcome = y[1:150], epsilon = 0.25, penalty=100)
#' }


SEMKL.regression=function(k,outcome,penalty,epsilon,tol=0.0001,max.iters=1000){
  n=length(outcome)
  A=c(rep(1,n),rep(-1,n))
  b=0
  r=0
  l=rep(0,2*n)
  c=c(epsilon-outcome,epsilon+outcome)
  u=rep(penalty,2*n)
  iters=0
  delta=rep(1,length(k))
  m=length(k)
  gamma=rep(1/m,m)
  gamma_all=list()
  #tic()
  while (max(delta)>tol && iters<max.iters){
    iters=iters+1
    gamma_all[[iters]]=gamma
    kk=Reduce('+',mapply("*", k, gamma,SIMPLIFY = FALSE))
    H=cbind(rbind(kk,-kk),rbind(-kk,kk))
    model=kernlab::ipop(c,H,A,b,l,u,r)
    var=list('alpha'= kernlab::primal(model)[1:n],'alphastar'= kernlab::primal(model)[(n+1):(2*n)] )
    fnorm=sapply(1:length(k), function(a){
      sqrt(gamma[a]^2*((var$alpha-var$alphastar)%*%k[[a]]%*%(var$alpha-var$alphastar)))})
    temp=gamma
    gamma=fnorm/sum(fnorm)
    delta=abs(temp-gamma)
  }
  # toc(log=TRUE,quiet=TRUE)
  #time=tic.log(format=FALSE)[[1]]$toc-tic.log(format=FALSE)[[1]]$tic
  w.ipop=(var$alpha-var$alphastar)%*%kk
  R=which(penalty-var$alpha>10^(-7)&var$alpha>10^(-7))
  Rstar=which(penalty-var$alphastar>10^(-7)&var$alphastar>10^(-7))
  b.low=(-epsilon-outcome+w.ipop)[Rstar]
  b.up=(epsilon-outcome+w.ipop)[R]
  b.ipop=as.numeric(names(which.max(table(round(union(b.low,b.up),6)))))
  results=list("alpha"=var$alpha, "alpha.star"=var$alphastar, "weight"=temp,
               "iters"=iters,'b'=b.ipop,'f'=w.ipop)
  return(results)
}
