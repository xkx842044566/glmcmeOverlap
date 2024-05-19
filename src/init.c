#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME:
 Check these declarations against the C/Fortran source code.
 */

/* .Call calls */
extern SEXP _glmcmeOverlap_cme_wls(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _glmcmeOverlap_cme_gaussian(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _glmcmeOverlap_mcp(SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"_glmcmenet_cme_wls", (DL_FUNC) &_glmcmeOverlap_cme_wls, 16},
  {"_glmcmenet_cme_gaussian", (DL_FUNC) &_glmcmeOverlap_cme_gaussian, 15},
  {"_glmcmenet_mcp", (DL_FUNC) &_glmcmeOverlap_mcp,  3},
  {NULL, NULL, 0}
};

void R_init_glmcmenet(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
