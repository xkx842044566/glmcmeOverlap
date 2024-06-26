# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

mcp <- function(beta, lambda, gamma) {
    .Call(`_glmcmeOverlap_mcp`, beta, lambda, gamma)
}

cme_wls <- function(XX, yy, family, K1, lambda_vec, gamma_vec, tau_vec, XX_sl, beta_vec, act_vec, multiplier, lambda_max, it_max, it_warm, reset, screen_ind) {
    .Call(`_glmcmeOverlap_cme_wls`, XX, yy, family, K1, lambda_vec, gamma_vec, tau_vec, XX_sl, beta_vec, act_vec, multiplier, lambda_max, it_max, it_warm, reset, screen_ind)
}

cme_gaussian <- function(XX, yy, K1, lambda_vec, gamma_vec, tau_vec, XX_sl, beta_vec, act_vec, multiplier, lambda_max, it_max, it_warm, reset, screen_ind) {
    .Call(`_glmcmeOverlap_cme_gaussian`, XX, yy, K1, lambda_vec, gamma_vec, tau_vec, XX_sl, beta_vec, act_vec, multiplier, lambda_max, it_max, it_warm, reset, screen_ind)
}

