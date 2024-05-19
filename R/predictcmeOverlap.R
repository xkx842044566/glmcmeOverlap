predictcmeOverlap <- function (fit.cme, newx,type=c("link", "response", "class"))
{
  dm <- dim(fit.cme$beta)
  eta <- array(NA, c(nrow(newx), dm[2], dm[3]))
  mu <- array(NA, c(nrow(newx), dm[2], dm[3]))
  for (i in 1:dm[2]) {
    for (j in 1:dm[3]) {
      eta[, i, j] <- fit.cme$intercept[i, j] + newx %*% matrix(fit.cme$beta[,
                                                                            i, j], ncol = 1)
      mu[, i, j] <- sapply(eta[, i, j],pfamily,fit.cme$family) # probability, mean of y, only for losgistic
    }
  }

  if(type=="response"){
    return(drop(mu))
  }else if(type=="link"){
    return(drop(eta))
  }else if(type=="class"){
    if (fit.cme$family=="binomial") {
      return(drop(1*(eta>0)))
    }else{
      stop("type='class' can only be used with family='binomial'", call.=FALSE)
    }
  }
}
