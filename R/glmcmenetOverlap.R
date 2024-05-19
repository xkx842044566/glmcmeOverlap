glmcmenetOverlap <- function(xme, xcme, y,
                             family=c("gaussian","binomial", "poisson"),
                             returnX.latent = FALSE, returnOverlap = FALSE,
                             lambda = exp(seq(from = log(max.lambda),to = log(max.lambda * 1e-06), length = 10)),
                             max.lambda = lambda0.cme(cbind(xme,xcme), y), gamma = 1/(0.125 - tau) + 0.001, tau = 0.01,
                             act.vec = rep(1,2*(ncol(xme) + ncol(xcme))), beta0 = rep(0, 2*(ncol(xme) + ncol(xcme))), penalty.factor=rep(1,2*ncol(xme)),
                             it.max = 250, screen_ind=F,str=F) {

  # Error checking
  # if (is.matrix(X)) {
  #   tmp <- try(X <- as.matrix(X), silent=TRUE)
  #   if (class(tmp)[1] == "try-error")  {
  #     stop("X must be a matrix or able to be coerced to a matrix")
  #   }
  # }
  # if (storage.mode(X)=="integer") X <- 1.0*X

  group <- grouplist(cbind(xme, xcme))

  incid.mat <- incidenceMatrix(cbind(xme, xcme), group) # group membership incidence matrix
  over.mat <- over.temp <- Matrix(incid.mat %*% t(incid.mat)) # overlap matrix
  grp.vec <- rep(1:nrow(over.mat), times = diag(over.mat)) # group index vector
  X.latent <- expandX(cbind(xme, xcme), group)

  diag(over.temp) <- 0
  if (all(over.temp == 0)) {
    cat("Note: There are NO overlaps between groups at all!", "\n")
    cat("      Now conducting non-overlapping group selection ...")
  }

  #XG <- newXG(X.latent, grp.vec, ncolY=1, bilevel=TRUE)
  K <- as.integer(table(grp.vec))
  K1 <- as.integer(if (min(grp.vec)==0) cumsum(K) else c(0, cumsum(K)))

  family <- match.arg(family)
  if (!is.double(penalty.factor)) penalty.factor <- as.double(penalty.factor)

  idx.const <- which(apply(X.latent, 2, function(xx) {
    all(xx == mean(xx))
  }))
  X.sc <- scale(X.latent, center = T, scale = T)
  X.sl <- attr(X.sc, "scaled:scale")
  X.ce <- attr(X.sc, "scaled:center")

  X.latent[, idx.const] <- 0
  X.sl[idx.const] <- 1

  if(family=="gaussian"){
    ret <- cme_gaussian(X.sc, y, K1, lambda, gamma,
                        tau, X.sl, beta0, act.vec, penalty.factor, max.lambda, it.max,
                        it_warm=3, reset=1, screen_ind)
  } else {
    ret <- cme_wls(X.sc, y, family, K1, lambda, gamma,
                    tau, X.sl, beta0, act.vec, penalty.factor, max.lambda, it.max,
                    it_warm=3, reset=1, screen_ind)
  }

  ##convert latent beta coefficients (gamma's) to non-latent beta's
  ret$beta <- array(0, dim=c(ncol(xme)+ncol(xcme),dim(ret$coefficients)[2:3]))
  for (i in 1:nrow(incid.mat)) {
    id <- which(incid.mat[i, ] == 1)
    ret$beta[id, , ] <- ret$beta[id, , ,drop = FALSE] + ret$coefficients[which(grp.vec == i), , , drop = FALSE]
  }
  ret$beta.lambda <- ret$beta[,,,drop = T]
  ret$intercept.lambda <- ret$intercept[,,drop = T]
  ret$xme <- xme
  ret$xcme <- xcme
  ret$y <- y
  ret$family <- family
  ret$incidence.mat <- incid.mat
  ret$group <- group
  ret$grp.vec <- grp.vec # this is 'group' argument in Package 'grpreg'
  if (returnX.latent) {
    ret$X.latent <- X.latent
  }
  if (returnOverlap) {
    ret$overlap.mat <- over.mat
  }

  val <- structure(ret, class = c('glmcmeOverlap', 'glmcme'))
  return(val)
}

