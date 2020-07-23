#toy example 10(2) features (same as in Deep Survival)
library(MASS)
library(survival)
library(gbm)
library(Rcpp)
library(randomForestSRC)
library(kernlab)
library(data.table)
library(ggpubr)
library(gridExtra)
library(RMKL)
#setwd("/home/chris/RMKL/R")


#sourceCpp('Coxdual.cpp')
#sourceCpp('kerngen.cpp')
#####linear######
n=1000
p=2
#beta=1
rho=0.5; #0.75
lambdaT=2  #baseline hazard =1/lambdaT
#lambdaC=50
#generate covaraince matrix  V=rho^|i-j|
#we can also generate R then use cholesky to intrdouce cor.
V=matrix(0,ncol=p,nrow=p)
for (i in 1:p) {
  for (j in 1:p ){
    V[i,j]=rho^abs(i-j)
  }
}

X=mvrnorm(n=n,mu=rep(0,p),Sigma=V)
Beta=c(1,2)

f.true=drop(X %*% Beta)
logh.true=log(lambdaT*exp(drop(X %*% Beta)))
T=-(log(runif(n)))/(lambdaT*exp(drop(X %*% Beta)))
#C=-(log(runif(n)))/(lambdaC*exp(drop(X %*% CBeta)))
C=rexp(n,rate=1/10)
obs.time<- pmin(T,C)
status <- T<=C
table(status)
#force censor
fi<-obs.time>10
obs.time[fi]=10
status[fi]= FALSE
table(status)

ordtr <- order(obs.time)
xx <- X[ordtr, ]
yy <- obs.time[ordtr]
del <- status[ordtr]

if (!del[1]) {
  first1 <- which(del)[1]
  xx <- xx[-(1:(first1 - 1)), ]
  yy <- yy[-(1:(first1 - 1))]
  del <- del[-(1:(first1 - 1))]
  nn <- n - first1 + 1
} else {
  nn <- n
}

rho0 <- .001*(del - seq(0, 10, length.out = nn))
klist <- list()
ktlist <- list()
klist[[1]] <- kernelMatrix(rbfdot(2), xx)
klist[[2]] <- kernelMatrix(vanilladot(), xx)
ktlist[[1]] <- kernelMatrix(rbfdot(2), xx, X)
ktlist[[2]] <- kernelMatrix(vanilladot(), xx, X)
kk <- simplify2array(klist)
kkk <- simplify2array(ktlist)
modmkl <- SurvMKL(y = yy, del = del, K = kk, rho = rho0, C = .5, lambda = .5, maxiter = 500, cri = .01)
premkl <- predict_Surv(modmkl,kkk)
survConcordance(Surv(obs.time, status) ~ premkl)
grid.arrange(grobs = plotlist, ncol = 3, bottom = text_grob(expression(x[1])), left = text_grob(expression(x[2])))
weight = apply(modmkl, 2, function(a) sqrt(sum(a^2)))
weight = weight/sum(weight)
weights

