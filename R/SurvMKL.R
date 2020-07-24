#' SurvMKL 
#'
#' This function conducts SurvMKL for precomputed gramm matrices
#' @param K The multiple kernel cube (3-d array)
#' @param y Vector of survival times
#' @param del Indicator vector of whether an event occured (0 = no event, 1 = event occured)
#' @param C cost parameter is the loss function
#' @param rho0 Argmin Dual objective function
#' @param lambda tuning parameter for the elastic net. Lambda closer 1 corresponds to L1 and closer to 0 corresponds to L2 penalties.
#' @param maxiter maximum number of allowed iteratons for outer loop, default to be 500
#' @param cri change between to iterations is smaller than this, algorithms is considered to have converged, default to be .001
#' @return alpha coeffiencents of the dual of SurvMKL
#' @useDynLib RMKL, .registration=TRUE
#' @importFrom Rcpp evalCpp 
#' @export
#' @examples
#' \dontrun{
#' library(RMKL)
#' library(kernlab)
#' data(Surv_data)
#' #Getting survival times in ascending order
#' ordtr <- order(Surv_data$time)
#' Surv_data_ordered = Surv_data[ordtr,]
#' 
#' xx = Surv_data_ordered[,1:2]
#' del = Surv_data_ordered$status
#' yy = Surv_data_ordered$time
#' if (!del[1]) {
#'   first1 <- which(del)[1]
#'   xx <- xx[-(1:(first1 - 1)), ]
#'   yy <- yy[-(1:(first1 - 1))]
#'   del <- del[-(1:(first1 - 1))]
#'   nn <- dim(Surv_data)[1] - first1 + 1
#' } else {
#'   nn <- dim(Surv_data)[1]
#' }
#' 
#' rho0 <- .001*(Surv_data$status - seq(0, 10, length.out = dim(Surv_data)[1]))
#' klist <- list(kernelMatrix(rbfdot(1), as.matrix(xx)),
#'               kernelMatrix(vanilladot(), as.matrix(xx)))
#' ktlist <- list(kernelMatrix(rbfdot(1), as.matrix(xx), as.matrix(Surv_data[,1:2])),
#'                kernelMatrix(vanilladot(), as.matrix(xx), as.matrix(Surv_data[,1:2])))
#' kk <- simplify2array(klist)
#' kkk <- simplify2array(ktlist)
#' modmkl <- SurvMKL(y = Surv_data$time, del = Surv_data$status, K = kk, rho =  
#' rho0, C = 0.005, lambda = 0.5, maxiter = 500, cri = .01)
#' }

SurvMKL <- function(K, y, del, rho0,C = .5, lambda = .5, maxiter = 500, cri =  0.01) {
     Coxdual(y0 = y, delta0 = del, k0 = K, rho0 = rho0, cc = C, lambda = lambda, 500, cri = cri)
 }



#' Predict SurvMKL
#' 
#' This function is used to predict SpicyMKL models
#' @param alpha coefficient
#' @param k0 the kernel cube needs prediction
#' @return The predicted score
#' @useDynLib RMKL
#' @importFrom Rcpp evalCpp
#' @export
#' @examples
#' \dontrun{
#' library(RMKL)
#' library(kernlab)
#' data(Surv_data)
#' #Getting survival times in ascending order
#' ordtr <- order(Surv_data$time)
#' Surv_data_ordered = Surv_data[ordtr,]
#' 
#' xx = Surv_data_ordered[,1:2]
#' del = Surv_data_ordered$status
#' yy = Surv_data_ordered$time
#' if (!del[1]) {
#'   first1 <- which(del)[1]
#'   xx <- xx[-(1:(first1 - 1)), ]
#'   yy <- yy[-(1:(first1 - 1))]
#'   del <- del[-(1:(first1 - 1))]
#'   nn <- dim(Surv_data)[1] - first1 + 1
#' } else {
#'   nn <- dim(Surv_data)[1]
#' }
#' 
#' rho0 <- .001*(Surv_data$status - seq(0, 10, length.out = dim(Surv_data)[1]))
#' klist <- list(kernelMatrix(rbfdot(1), as.matrix(xx)),
#'               kernelMatrix(vanilladot(), as.matrix(xx)))
#' ktlist <- list(kernelMatrix(rbfdot(1), as.matrix(xx), as.matrix(Surv_data[,1:2])),
#'                kernelMatrix(vanilladot(), as.matrix(xx), as.matrix(Surv_data[,1:2])))
#' kk <- simplify2array(klist)
#' kkk <- simplify2array(ktlist)
#' modmkl <- SurvMKL(y = Surv_data$time, del = Surv_data$status, K = kk, rho =  rho0, 
#' C = 0.005, lambda = 0.5, maxiter = 500, cri = .01)
#' mkl = predict_Surv(modmkl, kkk)
#' }

predict_Surv <- function(alpha, k0) {
  predictsurv(alpha, k0)
}


