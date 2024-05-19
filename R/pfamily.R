pfamily <- function (eta,family){
  if(family=="gaussian"){
    return(eta);
  }else if(family=="binomial"){
    if (eta > 16) {
      return(0.9999);
    } else if (eta < -16) {
      return(0.0001);
    } else {
      return(exp(eta)/(1+exp(eta)));
    }
  }else if(family=="poisson"){
    return(exp(eta));
  }
}
