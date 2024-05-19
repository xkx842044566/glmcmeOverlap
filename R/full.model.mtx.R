full.model.mtx <- function (xme)
{
  memtx <- xme
  nn <- nrow(memtx)
  pp <- ncol(memtx)
  colnret <- rep(NA, pp + 2 * pp * (pp - 1))
  ret <- matrix(NA, nrow = nn, ncol = 2 * pp * (pp - 1) +
                  pp)
  ininames <- rep(NA, pp)
  if (is.null(colnames(memtx))) {
    for (i in 1:pp) {
      ininames[i] <- paste0("V", i)
    }
    colnames(memtx) <- ininames
  }
  ret[, 1:pp] <- memtx
  colnret[1:pp] <- colnames(memtx)
  cmenames <- rep(NA, 2 * pp * (pp - 1))
  count <- pp + 1
  intmtx <- matrix(NA, nrow = 2 * pp * (pp - 1), ncol = 2)
  for (i in 1:pp) {
    for (j in 1:pp) {
      if (i != j) {
        ret[, count] <- memtx[, i] * as.numeric(memtx[,
                                                      j] >= 0)
        cmenames[count - pp] <- paste0(colnames(memtx)[i],
                                       "|", colnames(memtx)[j], "+")
        intmtx[count - pp, 1] <- i
        intmtx[count - pp, 2] <- j
        count <- count + 1
        ret[, count] <- memtx[, i] * as.numeric(memtx[,
                                                      j] < 0)
        cmenames[count - pp] <- paste0(colnames(memtx)[i],
                                       "|", colnames(memtx)[j], "-")
        intmtx[count - pp, 1] <- i
        intmtx[count - pp, 2] <- j
        count <- count + 1
      }
    }
  }
  colnret[(pp + 1):ncol(ret)] <- cmenames
  colnames(ret) <- colnret
  return(list(model.mtx = ret, cme.mtx = intmtx))
}
