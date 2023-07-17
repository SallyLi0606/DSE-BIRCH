library(fda)

fd_decomponent <- function(y, nharm) {
  # y shape: (num_samples, feat_dim)
  n <- dim(y)[2]
  t <- seq(0, 1, length.out=n)
  fdobj <- Data2fd(y=t(y), argvals=t)
  # FPCA
  fpca_result <- pca.fd(fdobj, nharm=nharm)
  varprop <- fpca_result$varprop
  # Extract Principal Components
  pc_base <- fpca_result$harmonics
  harmonics_prime <- deriv.fd(pc_base)
  C <- fpca_result$scores
  # Take the inner product of the derivative of the basis functions to obtain W
  W <- inprod(harmonics_prime, harmonics_prime)

  result = list(C=C, W=W, varprop=varprop)
  return(result)
}