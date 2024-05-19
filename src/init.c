#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME:
 Check these declarations against the C/Fortran source code.
 */

/* .Call calls */
extern SEXP _glmcmenet_cme_wls(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _glmcmenet_cme_gaussian(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _glmcmenet_cme_str(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _glmcmenet_mcp(SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"_glmcmenet_cme_wls", (DL_FUNC) &_glmcmenet_cme_wls, 17},
  {"_glmcmenet_cme_gaussian", (DL_FUNC) &_glmcmenet_cme_gaussian, 16},
  {"_glmcmenet_cme_str", (DL_FUNC) &_glmcmenet_cme_str, 18},
  {"_glmcmenet_mcp", (DL_FUNC) &_glmcmenet_mcp,  3},
  {NULL, NULL, 0}
};

void R_init_glmcmenet(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
