loss <- function(fitobj,y,yhat,n,family,type.measure=c("deviance","class","bic")){
  if("glmcmeOverlap" %in% class(fitobj)){
    selmat <- apply(fitobj$beta,c(2,3),function(x){return(length(which(x!=0)))})
  }else{
    selmat <- apply(fitobj$coefficients,c(2,3),function(x){return(length(which(x!=0)))})
  }
  if(type.measure=="deviance"){
    val <- array(NA, dim = dim(yhat))
    if (family=="gaussian") {
      val <- (y-yhat)^2
    } else if (family=="binomial") {
      val <- -2*(y*log(yhat)+(1-y)*log(1-yhat))
    }  else if (family=="poisson") {
      yly <- y*log(y)
      yly[y==0] <- 0
      val <- 2*(yly - y + yhat - y*log(yhat))
    }
  }else if(type.measure=="class"){
    val <- array(NA, dim = dim(yhat))
    if (family=="binomial") {
      val <- (yhat < 0.5) == y
    } else {
      val <- (y-yhat)^2
    }
  }else if(type.measure=="bic"){
    val <- array(NA, dim = dim(yhat))
    if (family=="gaussian") {
      val <- (y-yhat)^2+
        aperm(array(rep(selmat*log(n)/n, times=dim(yhat)[1]), dim(yhat)[c(2,3,1)]),c(3,1,2))
    } else if (family=="binomial") {
      val <- -2*(y*log(yhat)+(1-y)*log(1-yhat))+
        aperm(array(rep(selmat*log(n)/n, times=dim(yhat)[1]), dim(yhat)[c(2,3,1)]),c(3,1,2))
    }  else if (family=="poisson") {
      yly <- y*log(y)
      yly[y==0] <- 0
      val <- 2*(yly - y + yhat - y*log(yhat))+
        aperm(array(rep(selmat*log(n)/n, times=dim(yhat)[1]), dim(yhat)[c(2,3,1)]),c(3,1,2))
    }
    val <- replace(val, val == -Inf, 9999999)
  }
}



