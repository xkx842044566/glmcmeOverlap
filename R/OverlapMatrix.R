
## function: group of full model matrix
# ------------------------------------------------------------------------------
grouplist <- function(X) {

  names <- colnames(X)
  mainEffects <- unique(gsub("\\|.*$", "", names))

  groups <- list()

  # Initialize groups for each main effect
  for (effect in mainEffects) {
    groups[[paste0("S", effect)]] <- integer()
    groups[[paste0("C", effect)]] <- integer()
  }

  for (col in names) {
    if (grepl("\\|", col)) {
      # This is a conditional effect
      parts <- strsplit(col, "\\|")[[1]]
      mainEffect <- parts[1]
      conditionEffect <- gsub("\\+|-", "", parts[2])

      # Add to S(main) and C(condition)
      groups[[paste0("S", mainEffect)]] <- c(groups[[paste0("S", mainEffect)]], which(colnames(X) == col))
      groups[[paste0("C", conditionEffect)]] <- c(groups[[paste0("C", conditionEffect)]], which(colnames(X) == col))
    } else {
      # This is a main effect
      groups[[paste0("S", col)]] <- c(groups[[paste0("S", col)]], which(colnames(X) == col))
      groups[[paste0("C", col)]] <- c(groups[[paste0("C", col)]], which(colnames(X) == col))
    }
  }
  return(groups)
}
# ------------------------------------------------------------------------------



## function: overlap matrix c[i,j] = # of overlaps between group i and j.
##           Diagonals are group size. Provide interface for user to check
##           overlapping structure.
# ------------------------------------------------------------------------------
overlapMatrix <- function(X, group) {
  inc.mat <- incidenceMatrix(X, group)
  over.mat <- Matrix(inc.mat %*% t(inc.mat), sparse = TRUE, dimnames = dimnames(inc.mat))
  over.mat
}
# ------------------------------------------------------------------------------

## function: incidence matrix: I[i, j] = 1 if group i contains variable j.
# ------------------------------------------------------------------------------
incidenceMatrix <- function(X, group) {
  n <- nrow(X)
  pp <- ncol(X)
  if (! is.list(group)) {
    stop("Argument 'group' must be a list of integer indices or character names of variables!")
  }
  J <- length(group)
  grp.mat <- Matrix(0, nrow = J, ncol = pp, sparse = TRUE,
                    dimnames=list(as.character(rep(NA, J)),
                                  as.character(rep(NA, pp))))
  if(is.null(colnames(X))) {
    colnames(X) <- paste("V", 1:ncol(X), sep="")
  }
  if (is.null(names(group))) {
    names(group) <- paste("grp", 1:J, sep="")
  }

  if (is.numeric(group[[1]])) {
    for (i in 1:J) {
      ind <- group[[i]]
      grp.mat[i, ind] <- 1
      colnames(grp.mat)[ind] <- colnames(X)[ind]
    }
  } else { ## character, names of variables
    for (i in 1:J) {
      grp.i <- as.character(group[[i]])
      ind <- colnames(X) %in% grp.i
      grp.mat[i, ] <- 1*ind
      colnames(grp.mat)[ind] <- colnames(X)[ind]
    }
  }
  rownames(grp.mat) <- as.character(names(group))
  # check grp.mat
  if (all(grp.mat == 0)) {
    stop("The names of variables in X don't match with names in group!")
  }

  grp.mat
}
# ------------------------------------------------------------------------------



## function: expand design matrix X to overlapping design matrix (X.latent)
# -------------------------------------------------------------------------------
expandX <- function(X, group) {
  incidence.mat <- incidenceMatrix(X, group) # group membership incidence matrix
  over.mat <- Matrix(incidence.mat %*% t(incidence.mat), sparse = TRUE) # overlap matrix
  #grp.vec <- rep(1:nrow(over.mat), times = diag(over.mat)) # group index vector

  # expand X to X.latent
  X.latent <- NULL
  names <- NULL
  groupnm <- NULL

  ## the following code will automatically remove variables not included in 'group'
  for(i in 1:nrow(incidence.mat)) {
    idx <- incidence.mat[i,]==1
    X.latent <- cbind(X.latent, X[, idx, drop=FALSE])
    names <- c(names, colnames(incidence.mat)[idx])
    groupnm <- c(groupnm, rep(names(group)[i],diag(over.mat)[i]))
    #     colnames(X.latent) <- c(colnames(X.latent), colnames(X)[incidence.mat[i,]==1])
  }
  colnames(X.latent) <- paste(groupnm, '_', names, sep = "")
  X.latent
}
# -------------------------------------------------------------------------------




