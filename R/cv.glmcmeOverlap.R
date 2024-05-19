cv.glmcmeOverlap <- function (xme, xcme, y, family = c("gaussian","binomial", "poisson"), nfolds = 10, var.names = NULL, nlambda = 20,
                              lambda.min.ratio = 1e-06, ngamma = 20,
                              max.gamma = 150, ntau = 20, max.tau = 0.5, tau.min.ratio = 0.001,
                              it.max = 250, it.max.cv = 25, type.measure=c("deviance","class","bic"),
                              warm.str = c("NULL","lasso","adaptive_lasso","elastic","ncvreg","group_lasso","All"),penalty.factor=rep(1,2*ncol(xme)),
                              screen_ind=F,str=F)
{
  pme <- ncol(xme)
  pcme <- ncol(xcme)
  n <- nrow(xme)
  xmat <- cbind(xme, xcme)

  group <- grouplist(cbind(xme, xcme))
  incid.mat <- incidenceMatrix(cbind(xme, xcme), group) # group membership incidence matrix
  over.mat <- over.temp <- Matrix(incid.mat %*% t(incid.mat)) # overlap matrix
  grp.vec <- rep(1:nrow(over.mat), times = diag(over.mat)) # group index vector
  X.latent <- expandX(cbind(xme, xcme), group)
  #XG <- newXG(X.latent, grp.vec, m=penalty.factor, ncolY=1, bilevel=TRUE)
  K <- as.integer(table(grp.vec))
  K1 <- as.integer(if (min(grp.vec)==0) cumsum(K) else c(0, cumsum(K)))

  act.vec <- rep(-1, ncol(X.latent))
  if (warm.str == "lasso") {
    cvlas <- cv.glmnet(xmat, y,family = family,alpha=1,type.measure = "deviance")
    lasfit <- cvlas$glmnet.fit
    lasind <- which(lasfit$beta[, which(cvlas$lambda ==cvlas$lambda.min)] != 0)
    incid.mat.cum <- t(apply(incid.mat,1,cumsum))
    incidind <- which((matrix(incid.mat.cum[,lasind],ncol=length(lasind))*matrix(incid.mat[,lasind],ncol=length(lasind))) != 0, arr.ind = TRUE)
    lasind <- (incidind[, 1] - 1) * diag(over.mat)[1] +
      (matrix(incid.mat.cum[,lasind],ncol=length(lasind))*matrix(incid.mat[,lasind],ncol=length(lasind)))[incidind]
    act.vec[lasind] <- 1
    # for(l in c(2*lasind-1,2*lasind)){
    #  act.vec[(K1[l]+1):K1[l+1]] <- 1
    # }
  } else if (warm.str == "adaptive_lasso") {
    cv.ridge <- cv.glmnet(xmat,y, family=family, alpha=0)
    w3 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)
                       [, 1][2:(ncol(xmat)+1)]))^1 ## Using gamma = 1
    w3[w3[,1] == Inf] <- 999999999
    cvaplas <- cv.glmnet(xmat,y,family=family, alpha=1, type.measure="deviance", penalty.factor=w3)
    aplasfit <- cvaplas$glmnet.fit
    aplasind <- which(aplasfit$beta[, which(cvaplas$lambda ==cvaplas$lambda.min)] != 0)
    #for(l in c(2*aplasind-1,2*aplasind)){
    # act.vec[(K1[l]+1):K1[l+1]] <- 1
    #}
    incid.mat.cum <- t(apply(incid.mat,1,cumsum))
    incidind <- which((matrix(incid.mat.cum[,aplasind],ncol=length(aplasind))*matrix(incid.mat[,aplasind],ncol=length(aplasind))) != 0, arr.ind = TRUE)
    aplasind <- (incidind[, 1] - 1) * diag(over.mat)[1] +
      (matrix(incid.mat.cum[,aplasind],ncol=length(aplasind))*matrix(incid.mat[,aplasind],ncol=length(aplasind)))[incidind]
    act.vec[aplasind] <- 1
  }else if (warm.str == "elastic") {
    # control <- trainControl(method = "repeatedcv",
    #                         number = 5,
    #                         repeats = 5,
    #                         search = "random",
    #                         verboseIter = FALSE)
    # # Training ELastic Net Regression model
    # suppressWarnings(cvela <- train(y ~ .,
    #                     data = cbind(cbind(xme,xcme), y),
    #                     method = "glmnet",
    #                     family=family,
    #                     preProcess = c("center", "scale"),
    #                     tuneLength = 25,
    #                     trControl = control))
    #fitela <- glmnet(cbind(xme,xcme),y, family=family, alpha=cvela$bestTune$alpha,lambda = cvela$bestTune$lambda)#cv.elastic$finalModel
    cv.ela <- cv.glmnet(xmat,y, family=family, alpha=0.25,type.measure="deviance")#cv.elastic$finalModel
    fitela <- cv.ela$glmnet.fit
    elaind <- which(fitela$beta[,which(cv.ela$lambda==cv.ela$lambda.min)]!=0) #fitela$beta@i+1
    incid.mat.cum <- t(apply(incid.mat,1,cumsum))
    incidind <- which(incid.mat.cum[,elaind]*incid.mat[,elaind] != 0, arr.ind = TRUE)
    elaind <- (incidind[, 1] - 1) * diag(over.mat)[1] + (incid.mat.cum[,elaind]*incid.mat[,elaind])[incidind]
    act.vec[elaind] <- 1
    # for(l in c(2*elaind-1,2*elaind)){
    #  act.vec[(K1[l]+1):K1[l+1]] <- 1
    # }
  } else if (warm.str == "ncvreg") {
    cvncv <- cv.ncvreg(xmat,y,family = family,penalty="MCP")
    ncvfit <-cvncv$fit
    ncvind <- which(ncvfit$beta[,which(cvncv$lambda==cvncv$lambda.min)]!=0)[-1]-1
    incid.mat.cum <- t(apply(incid.mat,1,cumsum))
    incidind <- which((matrix(incid.mat.cum[,ncvind],ncol=length(ncvind))*matrix(incid.mat[,ncvind],ncol=length(ncvind))) != 0, arr.ind = TRUE)
    ncvind <- (incidind[, 1] - 1) * diag(over.mat)[1] +
      (matrix(incid.mat.cum[,ncvind],ncol=length(ncvind))*matrix(incid.mat[,ncvind],ncol=length(ncvind)))[incidind]
    act.vec[ncvind] <- 1
    # for(l in c(2*ncvind-1,2*ncvind)){
    #  act.vec[(K1[l]+1):K1[l+1]] <- 1
    # }
  } else if (warm.str == "group_lasso") {
    cvlas <- cv.glmnet(xme, y,family = family,alpha=1,type.measure = "deviance")
    lasfit <- cvlas$glmnet.fit
    lasind <- which(lasfit$beta[, which(cvlas$lambda ==cvlas$lambda.min)] != 0)
    # incid.mat.cum <- t(apply(incid.mat,1,cumsum))
    # incidind <- which((matrix(incid.mat.cum[,lasind],ncol=length(lasind))*matrix(incid.mat[,lasind],ncol=length(lasind))) != 0, arr.ind = TRUE)
    # lasind <- (incidind[, 1] - 1) * diag(over.mat)[1] +
    #    (matrix(incid.mat.cum[,lasind],ncol=length(lasind))*matrix(incid.mat[,lasind],ncol=length(lasind)))[incidind]
    # act.vec[lasind] <- 1
    for(l in c(2*lasind-1,2*lasind)){
      act.vec[(K1[l]+1):K1[l+1]] <- 1
    }
  } else if (warm.str == "All") {
    act.vec <- rep(1, ncol(X.latent))
  } else if (warm.str == "NULL") {
    act.vec <- rep(-1, ncol(X.latent))
  }


  min.tau <- max.tau * tau.min.ratio
  min.gamma <- max(max(apply(X.latent,2,function(x){8*nrow(X.latent)/sum(x^2)}))+ 0.001,1/(0.125 - min.tau) + 0.001)


  start_val <- get_start(X.latent, y,family)
  max.lambda <- start_val$lambda_max

  lambda <- exp(seq(from = log(max.lambda), to = log(max.lambda *
                                                       lambda.min.ratio), length = nlambda))

  gamma_vec = exp(seq(from = log(max.gamma), to = log(min.gamma),
                      length = ngamma - 1))
  gamma_vec = c(9.9e+35, gamma_vec)
  tau_vec = rev(exp(seq(from = log(max.tau), to = log(min.tau),
                        length = ntau)))

  # For each replicate ...

  # parms.min.bst <- c()
  # min.err <- 1e+30
  # cvm.gt.bst <- c()
  # cvm.lambda.bst <- c()
  parms1.min <- median(lambda)

  #if (!identical(sort(unique(y)), 0:1)) y <- as.double(y==max(y))
  ## Resample folds
  ## repeat{

  #foldid = sample(rep(seq(nfolds), length = n))
  if (nfolds < 3) {
    stop("nfolds must be bigger than 3; nfolds=10 recommended")
  }

  if (family=="binomial") {
    ind1 <- which(y==1)
    ind0 <- which(y==0)
    n1 <- length(ind1)
    n0 <- length(ind0)
    fold1 <- 1:n1 %% nfolds
    fold0 <- (n1 + 1:n0) %% nfolds
    fold1[fold1==0] <- nfolds
    fold0[fold0==0] <- nfolds
    foldid <- double(n)
    foldid[y==1] <- sample(fold1)
    foldid[y==0] <- sample(fold0)
  }else {
    foldid <- sample(1:n %% nfolds)
    foldid[foldid==0] <- nfolds
  }

  ## (1) Tuning gammas and tau:
  predmat = array(NA, c(n, length(tau_vec), length(gamma_vec)))

  #Perform N-fold CV
  print(paste0("Tuning gamma & tau: "))
  pb <- txtProgressBar(style = 3, width = floor(getOption("width")/2))
  for (i in seq(nfolds)) {
    setTxtProgressBar(pb, i/nfolds)
    which = (foldid == i)
    fitobj <- glmcmenetOverlap(xme = xme[!which, , drop = F], xcme = xcme[!which,, drop = F], y = y[!which],  family=family,
                               lambda = parms1.min, gamma = gamma_vec,
                               tau = tau_vec, act.vec = act.vec, penalty.factor=penalty.factor, max.lambda = max.lambda,
                               it.max = it.max.cv, screen_ind=F,str=str)
    xtest <- xmat[which, , drop = F]
    yhat <- predictcmeOverlap(fitobj, xtest, type="response")
    predmat[which, , ] <- loss(fitobj,y[which],yhat,n,family=family,type.measure=type.measure)

  }
  cat("\n")


  #Compute sum of cv deviation
  cvm.gt <- t(apply(predmat, c(2, 3), mean))
  whichmin = argmin(cvm.gt) #select tau and gamma
  ind2.min = whichmin
  parms2.min = c(gamma_vec[whichmin[1]], tau_vec[whichmin[2]])

  ## (2) Tuning lambdas:
  lambda <- exp(seq(from = log(max.lambda), to = log(max.lambda *
                                                       lambda.min.ratio), length = nlambda))
  predmat = array(NA, c(n, nlambda, 1))

  cvm <- matrix(NA, nlambda, 1)

  #Perform N-fold CV
  if (nfolds < 3) {
    stop("nfolds must be bigger than 3; nfolds=10 recommended")
  }
  print(paste0("Tuning lambdas: "))
  pb <- txtProgressBar(style = 3, width = floor(getOption("width")/2))
  for (i in seq(nfolds)) {
    setTxtProgressBar(pb, i/nfolds)
    which = (foldid == i)
    fitobj <- glmcmenetOverlap(xme = xme[!which, , drop = F], xcme = xcme[!which,
                                                                          , drop = F], y = y[!which], family=family,
                               lambda = lambda,gamma = parms2.min[1],
                               tau = parms2.min[2], act.vec = act.vec, penalty.factor=penalty.factor, max.lambda = max.lambda,
                               it.max = it.max.cv, screen_ind=screen_ind,str=str)
    xtest <- xmat[which, , drop = F]
    yhat <- predictcmeOverlap(fitobj, xtest, type="response")
    predmat[which, , ] <- loss(fitobj,y[which],yhat,n,family=family,type.measure=type.measure)
  }
  cat("\n")
  cvm.lambda <- apply(predmat, c(2, 3), mean)
  whichmin = argmin(cvm.lambda) #select lambdas
  ind1.min = whichmin
  parms1.min = lambda[whichmin[1]]
  parms.min <- c(parms1.min, parms2.min)

  ##Summarize into list
  obj = list(y = y, family=family, lambda = lambda,
             gamma = gamma_vec, tau = tau_vec, penalty.factor=penalty.factor,
             cvm.gt = cvm.gt, cvm.lambda = cvm.lambda)
  obj$params = parms.min
  names(obj$params) <- c("lambda", "gamma","tau")

  #Perform selection on full data with final parameters
  print(paste0("Fitting full data ..."))
  fitall <- glmcmenetOverlap(xme = xme, xcme = xcme, y,  family=family, lambda = lambda,
                             gamma = obj$params[2],
                             tau = obj$params[3], act.vec = act.vec, penalty.factor=penalty.factor, max.lambda = max.lambda,
                             it.max = it.max, screen_ind=screen_ind,str=F)
  obj$cme.fit <- fitall
  obj$select.idx <- which(fitall$beta[, which(lambda == obj$params[1]), ] !=
                            0)
  #if(length(obj$select.idx)>0){
  #  break
  #}
  #}

  obj$select.names <- var.names[obj$select.idx]


  #refit model based on selected variables
  #temp<-cbind(cbind(xme,xcme)[,obj$select.idx],y)
  #colnames(temp)<-c(obj$select.names,"y")
  #refit <- glm(y~.,temp,family="binomial")
  #obj$cme.refit <- refit
  #obj$class.err.rate<-mean(ifelse(refit$fitted.values>0.5,1,0)!=y)

  class(obj) = c("cv.glmcmeOverlap","cv.glmcme")
  return(obj)
}

