#' Prediction from MKL model
#'
#' This makes prediction for multiple kernel regression
#' 
#' @param ktest Gramm matrix of training data and test data
#' @param model MKL model 
#' @param outcome Outcome for the training data
#' @return yhat Predicted value for each test point
#' @return predicted Sign of yhat, which is the final predicted outcome 
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
#' model=SEMKL.regression(k=K$K.train, outcome = y[1:150], epsilon = 0.01, penalty=100)
#' pred=prediction.Regression(model=model, ktest=K$K.test,outcome=y)
#' plot(y[151:200],pred)
#' abline(a=0,b=1)
#' plot(x[151:200],y[151:200])
#' points(x[151:200],pred,col='red')
#' }


prediction.Regression=function(model,ktest,outcome){
  product=list()
  product=lapply(1:length(ktest), function(a) ktest[[a]]*model$weight[a])
  fushion=Reduce('+',product)
  yhat=(model$alpha-model$alpha.star)%*%t(fushion)-model$b
  return(yhat)
}
