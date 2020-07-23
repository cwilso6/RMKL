#' SurvMKL 
#'
#' This function conducts SurvMKL for precomputed gramm matrices
#' @param K The multiple kernel cube (3-d array)
#' @param y Vector of survival times
#' @param del Indicator vector of whether an event occured (0 = no event, 1 = event occured)
#' @param C cost parameter is the loss function
#' @param lambda tuning parameter for the elastic net. Lambda closer 1 corresponds to L1 and closer to 0 corresponds to L2 penalties.
#' @param maxiter maximum number of allowed iteratons for outer loop, default to be 500
#' @param cri change between to iterations is smaller than this, algorithms is considered to have converged, default to be .001
#' @param InnerMaxiter maximum number of allowed iteratons for inner loop, default to be 500
#' @return alpha coeffiencents of the dual of SurvMKL
#' @useDynLib RMKL, .registration=TRUE
#' @importFrom Rcpp evalCpp 
#' @export
SurvMKL <- function(K, y, del, rho0,C = .5, lambda = .5, maxiter = 500, cri = 0.01) {
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


predict_Surv <- function(alpha, k0) {
  predictsurv(alpha, k0)
}

