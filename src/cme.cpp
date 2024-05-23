// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <iostream>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace Rcpp;

//Compares two doubles for equality
bool dbleq(double a, double b)
{
  return (fabs(a - b) < DBL_EPSILON);
}


// Computate max absolute value
double fmax2 (double x, double y){
  double ret = 0.0;
  if (fabs(x)>fabs(y)){
    ret=x;
  }else{
    ret=y;
  }
  return (ret);
}

// Max of x
double max(vector<double>& x, int n) {
  double val = x[0];
  for (int i=1; i<n; i++) {
    if (x[i] > val) val = x[i];
  }
  return(val);
}


// Pr(y=1) for binomial
double pbinomial(double eta) {
  if (eta > 16) {
    return(0.9999);
  } else if (eta < -16) {
    return(0.0001);
  } else {
    return(exp(eta)/(1+exp(eta)));
  }
}


// Cross product of y with jth column of X
double crossprod(vector<double>& X, vector<double>& yy, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += X[nn+i]*yy[i];
  return(val);
}

// Weighted cross product of y with jth column of x
double wcrossprod(vector<double>& X, vector<double>& yy, vector<double>& w, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += X[nn+i]*yy[i]*w[i];
  return(val);
}

// Weighted sum of squares of jth column of X
double wsqsum(vector<double>&X, vector<double>&w, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += w[i] * pow(X[nn+i], 2);
  return(val);
}

// Sum of squares of jth column of X
double sqsum(vector<double>&X, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += pow(X[nn+i], 2);
  return(val);
}

double sum(vector<double>&x, int n) {
  double val=0;
  for (int i=0;i<n;i++) val += x[i];
  return(val);
}

int sum_act(vector<double>&x, vector<int>& indices) {
  int val=0;
  for(int index : indices) {
    if(index >= 0 && index < x.size()) {
      val += x[index];
    } else {
      cerr << "Warning: Index " << index << " out of bounds. Ignored." << endl;
    }
  }
  return(val);
}


//Threshold function for ME (varying lambda)
double s_me(double inprod, double v, double& lambda, double gamma, double& delta){

  // inprod - inner product to threshold
  // lambda - penalties for reg, sib, cou and inv (ignore inv for now)
  // gamma - assumed fixed
  // delta - linearized penalties for reg, sib, cou and inv (ignore inv for now)
  // nn - the number of observations, n

  //Compute thresholds
  double ratio = delta/lambda;
  double ret = 0.0;
  double sgn = 0.0;
  if (inprod < 0.0){
    sgn = -1.0;
  }
  else{
    sgn = 1.0;
  }

  if (abs(inprod) < v*lambda*gamma ){
    if (abs(inprod) >= delta ){
      ret = ( abs(inprod)-delta ) / (1 - 1.0/gamma*ratio);
    }
    else{
      ret = 0.0;
    }
  }
  else{
    ret = abs(inprod);
  }
  ret=ret/v;
  return (sgn*ret);

}


double s_me_gaussian(double inprod, double& lambda, double gamma, double& delta){

  // inprod - inner product to threshold
  // lambda - penalties for reg, sib, cou and inv (ignore inv for now)
  // gamma - assumed fixed
  // delta - linearized penalties for reg, sib, cou and inv (ignore inv for now)
  // nn - the number of observations, n

  //Compute thresholds
  double ratio = delta/lambda;
  double ret = 0.0;
  double sgn = 0.0;
  if (inprod < 0.0){
    sgn = -1.0;
  }
  else{
    sgn = 1.0;
  }

  if (abs(inprod) < lambda*gamma ){
    if (abs(inprod) >= delta ){
      ret = ( abs(inprod)-delta ) / (1 - 1.0/gamma*ratio);
    }
    else{
      ret = 0.0;
    }
  }
  else{
    ret = abs(inprod);
  }

  return (sgn*ret);

}


//MCP penalty
// [[Rcpp::export]]
double mcp(double beta, double& lambda, double& gamma){
  double ret = 0.0;
  if (abs(beta) <= (lambda*gamma) ){
    ret = abs(beta) - pow(beta,2.0)/(2.0*lambda*gamma);
  }
  else{
    ret = lambda*gamma/(2.0);
  }
  return(ret);
}

//KKT condition
bool kkt(double inprod,  double& cur_delta){
  //Checks KKT condition for \beta=0.0
  bool ret;
  double lb = -inprod - cur_delta;
  double ub = -inprod + cur_delta;
  if ((0.0 >= lb)&&(0.0 <= ub)){
    ret = true; //kkt satisfied
  }
  else{
    // cout << "lb: " << lb << ", ub: " << ub << endl;
    ret = false;
  }
  return(ret);
}



//One run of coordinate descent under default structure
bool coord_des_onerun_wls(int pme, int nn, NumericVector& K1,
                              double& lambda, double& cur_delta,
                              bool dummy, double tau, double gamma,
                              vector<double>& X, NumericVector& yy,
                              CharacterVector& family,
                              vector<double>& delta,
                              vector<bool>& act, double& inter,
                              vector<double>& beta, vector<double>& mg,
                              vector<double>& eta, vector<double>& resid, vector<double>& W, double dev){
  bool chng_flag = false;
  double cur_beta = 0.0;
  double cur_lambda = 0.0;
  double cur_inter = 0.0;
  double xwr = 0.0;
  double xwx = 0.0;
  double inprod = 0.0;
  double v = 0.0;
  double mu = 0.0;
  int J = K1.size() - 1;

  //double dev = 0.0;

  // Extract the family type from CharacterVector
  string familyType = Rcpp::as<string>(family[0]);


  if (familyType == "binomial") {
    v = 0.25;
    for (int i=0; i<nn; i++) {
      mu = pbinomial(eta[i]);
      W[i] = fmax2(mu*(1-mu), 0.0001);
      resid[i] = (yy[i] - mu)/W[i];
      if (yy[i]==1) dev = dev - log(mu);
      if (yy[i]==0) dev = dev - log(1-mu);
    }
  } else if (familyType == "poisson") {
    v = exp(max(eta, nn));
    for (int i=0; i<nn; i++) {
      mu = exp(eta[i]);
      W[i] = mu;
      resid[i] = (yy[i] - mu)/W[i];
      if (yy[i]!=0) dev += yy[i]*log(yy[i]/mu);
    }
  }


  //Update intercept
  cur_inter= inter;

  xwr = crossprod(W, resid, nn, 0);
  xwx = sum(W,nn);
  inter = xwr/xwx + cur_inter;
  for (int i=0; i<nn; i++) {
    resid[i] -= inter - cur_inter;
    eta[i] += inter - cur_inter;
    // if (familyType == "binomial"){
    //   mu = pbinomial(eta[i]) ;
    //   W[i] = fmax2(mu*(1-mu),0.0001);
    // }else if(familyType == "poisson"){
    //   mu = exp(eta[i]) ;
    //   W[i] = mu;
    // }
  }

  for (int g=0; g<J; g++) {

    //if (act_gr[g]){
    if (act[K1[g]]){
      cur_beta = beta[K1[g]];

      //int K = K1[g+1] - K1[g];
      xwr = wcrossprod(X, resid, W, nn, K1[g]);
      xwx = wsqsum(X, W, nn, K1[g]);
      v = xwx/((double)nn);
      // inprod = inprod/((double)nn)+beta_me[K1[g]]; i.e, zj in proof
      inprod = xwr/((double)nn)+v*beta[K1[g]]; //checked to pod from update eqn (mod from above eqn)

      //int gind = floor((double)g/2.0);
      cur_delta = delta[g];
      cur_lambda = lambda*mg[g];

      //Perform ME thresholding
      beta[K1[g]] = s_me(inprod,v,cur_lambda,gamma,cur_delta);

      // Update eta, mu, weight and delta
      if (!dbleq(beta[K1[g]],cur_beta)){ // if beta changed...

        //Update resid eta, mu, weight
        for (int k=0;k<nn;k++){
          resid[k] -= X[K1[g]*nn+k]*(beta[K1[g]]-cur_beta);
          eta[k] += X[K1[g]*nn+k]*(beta[K1[g]]-cur_beta);
          // if (familyType == "binomial"){
          //   mu = pbinomial(eta[k]) ;
          //   W[k] = fmax2(mu*(1-mu),0.0001);
          // }else if(familyType == "poisson"){
          //   mu = exp(eta[k]) ;
          //   W[k] = mu;
          // }
          //v += (X_me[j*nn+k]*W[k]*X_me[j*nn+k])/((double)nn);
        }
        // xwx = wsqsum(X_me, W, nn, j);
        // v = xwx/((double)nn);

        //Update deltas
        double offset = mcp(beta[K1[g]],cur_lambda,gamma)-mcp(cur_beta,cur_lambda,gamma); // new - old
        delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

        chng_flag = true;
      }
    }


    //CD for covariates in group g
    for (int j=K1[g]+1;j<K1[g+1];j++){

      //Only update if active
      if (act[j]){
        cur_beta = beta[j];

        // Updata covariates
        xwr = wcrossprod(X, resid, W, nn, j);
        xwx = wsqsum(X, W, nn, j);
        v = xwx/((double)nn);
        // inprod = inprod/((double)nn)+beta_me[j]; i.e, zj in proof
        inprod = xwr/((double)nn)+v*beta[j]; //checked to pod from update eqn (mod from above eqn)

        //string me = Rcpp::as<string>(names_me[j]);
        //int delta_ind = findind(effectIndexMap,me);

        //Update cur_delta
        //int gind = floor((double)g/2.0);
        cur_delta = delta[g];
        cur_lambda = lambda*mg[g];

        //Perform ME thresholding
        beta[j] = s_me(inprod,v,cur_lambda,gamma,cur_delta);

        // Update eta, mu, weight and delta
        if (!dbleq(beta[j],cur_beta)){ // if beta changed...

          //Update resid eta, mu, weight
          for (int k=0;k<nn;k++){
            resid[k] -= X[j*nn+k]*(beta[j]-cur_beta);
            eta[k] += X[j*nn+k]*(beta[j]-cur_beta);
            // if (familyType == "binomial"){
            //   mu = pbinomial(eta[k]) ;
            //   W[k] = fmax2(mu*(1-mu),0.0001);
            // }else if(familyType == "poisson"){
            //   mu = exp(eta[k]) ;
            //   W[k] = mu;
            // }
            //v += (X_me[j*nn+k]*W[k]*X_me[j*nn+k])/((double)nn);
          }
          // xwx = wsqsum(X_me, W, nn, j);
          // v = xwx/((double)nn);

          //Update deltas
          double offset = mcp(beta[j],cur_lambda,gamma)-mcp(cur_beta,cur_lambda,gamma); // new - old
          delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;


          //Update flag
          chng_flag = true;

        }

        //Reduce A|B+ and A|B- to A
        //
        if (abs(beta[j]) > 0.0){ //if current CME is active
          int id = j - K1[g];
          if (id % 2 == 1){ //cme is .|.+
            if (abs(beta[j+1]) > 0.0){ //if cme .|.- is also in model...

              double chg, cur_beta_me, cur_beta_cme1, cur_beta_cme2;

              if ( abs(beta[j]) > abs(beta[j+1]) ){// if abs(.|.+) > abs(.|.-)
                chg = beta[j+1]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j+1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j] -= chg; // update larger CME
                beta[j+1] = 0.0; // remove smaller CME
              }else{// if abs(.|.+) < abs(.|.-)
                chg = beta[j]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j+1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j+1] -= chg; // update larger CME
                beta[j] = 0.0; // remove smaller CME
              }

              //Update deltas and flag
              double offset = mcp(beta[K1[g]],cur_lambda,gamma)-mcp(cur_beta_me,cur_lambda,gamma); // new - old (for me)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j],cur_lambda,gamma)-mcp(cur_beta_cme1,cur_lambda,gamma);
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j+1],cur_lambda,gamma)-mcp(cur_beta_cme2,cur_lambda,gamma); // new - old (for .|.-)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;
              //residuals shouldn't change

            }
          }else{ //cme is .|.-
            if (abs(beta[j-1]) > 0.0){ //if cme .|.+ is also in model...

              double chg, cur_beta_me, cur_beta_cme1, cur_beta_cme2;

              if ( abs(beta[j]) > abs(beta[j-1]) ){// if abs(.|.+) < abs(.|.-)
                chg = beta[j-1]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j-1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j] -= chg; // update larger CME
                beta[j-1] = 0.0; // remove smaller CME
              }else{// if abs(.|.+) > abs(.|.-)
                chg = beta[j]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j-1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j-1] -= chg; // update larger CME
                beta[j] = 0.0; // remove smaller CME
              }

              //Update deltas and flag
              double offset = mcp(beta[K1[g]],cur_lambda,gamma)-mcp(cur_beta_me,cur_lambda,gamma); // new - old (for me)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j],cur_lambda,gamma)-mcp(cur_beta_cme1,cur_lambda,gamma); // new - old (for .|.+)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j-1],cur_lambda,gamma)-mcp(cur_beta_cme2,cur_lambda,gamma);
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              //residuals shouldn't change

            }
          }
        }
      }
    }
    //}
  }


  return (chng_flag);

}


bool coord_des_onerun_gaussian(int pme, int nn, NumericVector& K1,
                               double& lambda, double& cur_delta,
                               bool dummy, double tau, double gamma,
                               vector<double>& X, NumericVector& yy,
                               vector<double>& delta,
                               vector<bool>& act, double& inter,
                               vector<double>& beta, vector<double>& mg,
                               vector<double>& resid){
  bool chng_flag = false;
  double cur_beta = 0.0;
  double cur_lambda = 0.0;
  double inprod = 0.0;
  int J = K1.size() - 1;


  //cur_inter= inter;

  for (int g=0; g<J; g++) {

    //if (act_gr[g]){
    if (act[K1[g]]){

      cur_beta = beta[K1[g]];

      //Compute inner product
      inprod = 0.0;
      for (int k=0;k<nn;k++){
        inprod += (resid[k]*X[K1[g]*nn+k]);
      }
      // inprod = inprod/((double)nn)+beta_me[j];
      inprod = inprod/((double)nn)+(((double)nn)-1)/((double)nn)*beta[K1[g]]; //checked to pod from update eqn (mod from above eqn)

      //int gind = floor((double)g/2.0);
      cur_delta = delta[g];
      cur_lambda = lambda*mg[g];


      //Perform ME thresholding
      beta[K1[g]] = s_me_gaussian(inprod,cur_lambda,gamma,cur_delta);

      // Update eta, mu, weight and delta
      if (!dbleq(beta[K1[g]],cur_beta)){ // if beta changed...

        //Update resid eta, mu, weight
        for (int k=0;k<nn;k++){
          resid[k] = resid[k] - X[K1[g]*nn+k]*(beta[K1[g]]-cur_beta);
        }

        //Update deltas
        double offset = mcp(beta[K1[g]],cur_lambda,gamma)-mcp(cur_beta,cur_lambda,gamma); // new - old
        delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

        chng_flag = true;
      }
    }

    cur_beta = 0.0;
    inprod = 0.0;
    //CD for covariates in group g
    for (int j=K1[g]+1;j<K1[g+1];j++){

      //Only update if active
      if (act[j]){
        cur_beta = beta[j];

        inprod = 0.0;
        for (int l=0;l<nn;l++){
          inprod += (resid[l]*X[j*nn+l]);
        }
        inprod = inprod/((double)nn)+(((double)nn)-1)/((double)nn)*beta[j];

        //Update cur_delta
        //int gind = floor((double)g/2.0);
        cur_delta = delta[g];
        cur_lambda = lambda*mg[g];


        //Perform ME thresholding
        beta[j] = s_me_gaussian(inprod,cur_lambda,gamma,cur_delta);

        // Update eta, mu, weight and delta
        if (!dbleq(beta[j],cur_beta)){ // if beta changed...

          //Update resid eta, mu, weight
          for (int ll=0;ll<nn;ll++){
            resid[ll] = resid[ll] - X[j*nn+ll]*(beta[j]-cur_beta);
          }

          //Update deltas
          double offset = mcp(beta[j],cur_lambda,gamma)-mcp(cur_beta,cur_lambda,gamma);
          delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

          //Update flag
          chng_flag = true;

        }

        //Reduce A|B+ and A|B- to A
        //
        if (abs(beta[j]) > 0.0){ //if current CME is active
          int id = j - K1[g];
          if (id % 2 == 1){ //cme is .|.+
            if (abs(beta[j+1]) > 0.0){ //if cme .|.- is also in model...

              double chg, cur_beta_me, cur_beta_cme1, cur_beta_cme2;

              if ( abs(beta[j]) > abs(beta[j+1]) ){// if abs(.|.+) > abs(.|.-)
                chg = beta[j+1]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j+1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j] -= chg; // update larger CME
                beta[j+1] = 0.0; // remove smaller CME
              }else{// if abs(.|.+) < abs(.|.-)
                chg = beta[j]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j+1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j+1] -= chg; // update larger CME
                beta[j] = 0.0; // remove smaller CME
              }

              //Update deltas and flag
              double offset = mcp(beta[K1[g]],cur_lambda,gamma)-mcp(cur_beta_me,cur_lambda,gamma); // new - old (for me)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset));

              offset = mcp(beta[j],cur_lambda,gamma)-mcp(cur_beta_cme1,cur_lambda,gamma);
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j+1],cur_lambda,gamma)-mcp(cur_beta_cme2,cur_lambda,gamma); // new - old (for .|.-)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;
              //residuals shouldn't change

            }
          }else{ //cme is .|.-
            if (abs(beta[j-1]) > 0.0){ //if cme .|.+ is also in model...

              double chg, cur_beta_me, cur_beta_cme1, cur_beta_cme2;

              if ( abs(beta[j]) > abs(beta[j-1]) ){// if abs(.|.+) < abs(.|.-)
                chg = beta[j-1]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j-1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j] -= chg; // update larger CME
                beta[j-1] = 0.0; // remove smaller CME
              }else{// if abs(.|.+) > abs(.|.-)
                chg = beta[j]; // change
                cur_beta_me = beta[K1[g]]; // current beta me
                cur_beta_cme1 = beta[j]; // current beta cme 1
                cur_beta_cme2 = beta[j-1]; // current beta cme 2
                beta[K1[g]] += chg; // update ME with smaller CME
                beta[j-1] -= chg; // update larger CME
                beta[j] = 0.0; // remove smaller CME
              }

              //Update deltas and flag
              double offset = mcp(beta[K1[g]],cur_lambda,gamma)-mcp(cur_beta_me,cur_lambda,gamma); // new - old (for me)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j],cur_lambda,gamma)-mcp(cur_beta_cme1,cur_lambda,gamma); // new - old (for .|.+)
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;

              offset = mcp(beta[j-1],cur_lambda,gamma)-mcp(cur_beta_cme2,cur_lambda,gamma);
              delta[g] = delta[g] * (exp(-(tau/cur_lambda) * offset)) ;


              //residuals shouldn't change

            }
          }
        }
      }
    }
    //}
  }


  return (chng_flag);

}




// [[Rcpp::export]]
List cme_wls(NumericMatrix& XX, NumericVector& yy, CharacterVector& family,
              NumericVector& K1,
              NumericVector& lambda_vec,
              NumericVector& gamma_vec, NumericVector& tau_vec,
              NumericVector& XX_sl, NumericVector& beta_vec, NumericVector& act_vec, NumericVector& multiplier,
              double lambda_max, int it_max, int it_warm, int reset, bool screen_ind) {
  // // [[Rcpp::plugins(openmp)]]
  //------------------------------------------------------------
  // XX - Full model matrix including both ME and CME effects (assume normalized)
  // yy - Response vector of length nn
  // family- family of GLM
  // lambda_sib_vec - Vector of sibling penalties (decr. sequence)
  // lambda_cou_vec - Vector of cousin penalties (decr. sequence)
  // tau - Exponential penalty parameter
  // gamma - MC+ non-convex penalty parameter
  // beta_vec - Initial beta value
  // it_max - Maximum iterations for coordinate descent
  //------------------------------------------------------------

  //Variable initialization
  int pp = XX.ncol(); //# of CMEs
  int nn = XX.nrow(); //# of observations
  int nlambda = lambda_vec.size();
  int it_inner = 0;
  int it_max_reset = it_max / reset;
  bool cont = true;
  bool chng_flag = false;
  //double v = 0;
  double mu = 0;
  int J = K1.size() - 1;
  int pme = J/2; //# of MEs

  // Extract the family type from CharacterVector
  string familyType = Rcpp::as<string>(family[0]);

  //Vectorize model matrices
  vector<double> X(nn*pp); //for ME
  //vector<double> X_cme(nn*pcme); //for CME
  for (int i=0;i<pp;i++){
    for (int j=0;j<nn;j++){
      X[i*nn+j] = XX(j,i);
    }
  }

  //Check whether lambda is to be iterated or not
  bool lambda_it;
  int niter_1; //Number to iterate first
  int niter_2; //Number to iterate next
  if (gamma_vec.size()>1){ //Iterate on gamma and tau
    lambda_it = false;
    // niter_1 = gamma_vec.size(); //ch
    // niter_2 = tau_vec.size();
    niter_1 = tau_vec.size();
    niter_2 = gamma_vec.size();
  }
  else{
    lambda_it = true;
    niter_1 = nlambda;
    niter_2 = 1;
  }

  //Containers for beta and active set (alpha)
  arma::cube beta_cube(pp,niter_1,niter_2); //betas to return
  arma::cube delta_cube(J,niter_1,niter_2); //deltas to return
  arma::mat nz(niter_1,niter_2);
  arma::mat inter_mat(niter_1,niter_2); //intercept to return
  arma::mat dev_mat(niter_1,niter_2); //deviation to return
  arma::mat beta_mat(pp,niter_1);
  arma::mat delta_mat(J,niter_1);

  vector<double> beta(pp,0.0); //for MEs
  for (int i=0;i<pp;i++){
    beta[i] = beta_vec[i];
  }


  //Set all factors as active to begin
  vector<bool> act(pp,true); //Current active set
  //vector<bool> act_cme(pcme,true);
  vector<bool> scr(pp,true); //Screened active set
  //vector<bool> scr_cme(pcme,true);
  //bool kkt_bool;

  vector<double> mg(J,1);
  //vector<double> m_cme(pcme,1);
  for (int i=0;i<J;i++){
    mg[i] = multiplier[i];
  }
  // for (int i=0;i<pcme;i++){
  //   m_cme[i] = multiplier[pme+i];
  // }

  //Containers for linearized slopes Delta
  vector<double> delta(J); //Linearized penalty for siblings/cousins (sib(A), cou(A), sib(B), cou(B),...)
  double lambda; //Current penalties
  double cur_delta; //Current delta vector
  double cur_lambda;
  double gamma;
  double tau;
  lambda = lambda_vec[0];
  gamma = gamma_vec[0];
  tau = tau_vec[0];

  //Update resid (eta)
  vector<double> eta(nn);
  vector<double> W(nn);
  vector<double> resid(nn); //Residual vector
  arma::cube resid_cube(nn,niter_1,niter_2);
  arma::mat resid_mat(nn,niter_1);
  arma::cube scr_cube(pp,niter_1,niter_2); //screening vector
  arma::mat scr_mat(pp,niter_1);

  double nullDev = 0.0;
  double dev = 0.0;
  double inter= 0.0; //for intercept
  //double inprod = 0.0; //inner product
  double cj = 0.0;
  double vj = 0.0;
  double thresh = 0.0; //threshold for screening
  //int size = 0;
  int num_act = 0;
  int num_scr = 0;

  double ymean = 0.0;
  for (int i=0;i<nn;i++){
    ymean += (1.0/(double)nn)*yy[i];
  }
  if (familyType == "binomial") {
    inter = log(ymean/(1-ymean));
    for (int i=0; i<nn; i++) {
      eta[i] = log(ymean/(1-ymean)); //
      mu = ymean ;
      W[i] = fmax2(mu*(1-mu),0.0001);
      resid[i] = (yy[i]-mu)/W[i];
      nullDev -= 2*yy[i]*log(ymean) + 2*(1-yy[i])*log(1-ymean);
      dev = nullDev;
    }
  } else if (familyType == "poisson") {
    inter = log(ymean);
    for (int i=0;i<nn;i++) {
      eta[i] = log(ymean);
      mu = ymean ;
      W[i] = mu;
      resid[i] = (yy[i]-mu)/W[i];
      if (yy[i]!=0) nullDev += 2*(yy[i]*log(yy[i]/ymean) + ymean - yy[i]);
      else nullDev += 2*ymean;
      dev = nullDev;
    }
  }

  //vector<bool> kkt_v_me(pme,true);
  //vector<bool> kkt_v_cme(pcme,true); //KKT checks

  // Optimize for each penalty combination
  // #pragma omp parallel for
  for (int b=0; b<niter_2; b++){ //iterate over cousins...

    for (int a=0; a<niter_1; a++){ //iterate over siblings...

      // cout << "Tuning ... a = " << a << ", b = " << b << endl;//Update iteration variables
      if (lambda_it){
        lambda = lambda_vec[a];
      }
      else{
        tau = tau_vec[a];
        gamma = gamma_vec[b];
      }

      //Return trivial solution of \beta=0 when \lambda_s + \lambda_c >= \lambda_max
      // if ( (lambda[0]+lambda[1]) >= lambda_max){
      if ( (a==0) || ( lambda>= lambda_max) ){
        for (int i=0;i<pp;i++){//reset beta
          beta[i] = 0.0;
        }

        if (familyType == "binomial") {
          inter = log(ymean/(1-ymean));
          for (int i=0; i<nn; i++) {
            eta[i] = log(ymean/(1-ymean)); //
            mu = ymean ;
            W[i] = fmax2(mu*(1-mu),0.0001);
            resid[i] = (yy[i]-mu)/W[i];
            dev -= 2*yy[i]*log(ymean) + 2*(1-yy[i])*log(1-ymean);
          }
        } else if (familyType == "poisson") {
          inter = log(ymean);
          for (int i=0;i<nn;i++) {
            eta[i] = log(ymean);
            mu = ymean ;
            W[i] = mu;
            resid[i] = (yy[i]-mu)/W[i];
            if (yy[i]!=0) dev += 2*(yy[i]*log(yy[i]/ymean) + ymean - yy[i]);
            else dev += 2*ymean;
          }
        }
        num_act = 0;
        num_scr = 0;
        for (int i=0;i<pp;i++){//reset active flag
          act[i] = true;
          scr[i] = true;
          num_act ++;
          num_scr ++;
        }
        // for (int i=0;i<pcme;i++){
        //   act_cme[i] = true;
        //   scr_cme[i] = true;
        //   num_act ++;
        //   num_scr ++;
        // }
        //cout << "num_act: " << num_act << endl;
        if ( lambda >= lambda_max){
          goto cycend;
        }
      }

      // // RESET AFTER EACH RUN
      for (int i=0;i<pp;i++){//reset beta
        beta[i] = 0.0;
      }

      if (familyType == "binomial") {
        inter = log(ymean/(1-ymean));
        for (int i=0; i<nn; i++) {
          eta[i] = log(ymean/(1-ymean)); //
          mu = ymean ;
          W[i] = fmax2(mu*(1-mu),0.0001);
          resid[i] = (yy[i]-mu)/W[i];
          dev -= 2*yy[i]*log(ymean) + 2*(1-yy[i])*log(1-ymean);
        }
      } else if (familyType == "poisson") {
        inter = log(ymean);
        for (int i=0;i<nn;i++) {
          eta[i] = log(ymean);
          mu = ymean ;
          W[i] = mu;
          resid[i] = (yy[i]-mu)/W[i];
          if (yy[i]!=0) dev += 2*(yy[i]*log(yy[i]/ymean) + ymean - yy[i]);
          else dev += 2*ymean;
        }
      }

      //Recompute deltas
      fill(delta.begin(),delta.end(),lambda);
      for (int g=0; g<J; g++) {
        cur_lambda = lambda*mg[g];
        //fill(delta.begin(),delta.end(),cur_lambda); //assigns each element the value lambda
        //delta[g] = cur_lambda;

        for (int j=K1[g];j<K1[g+1];j++){

          //int gind = floor((double)g/2.0); //index for sibling or cousin group
          delta[g] = delta[g] * mg[g] * (exp(-(tau/cur_lambda) * mcp(beta[j],cur_lambda,gamma) )) ;
        }
      }

      //Coordinate descent with warm active set resets
      for (int q=0; q<reset; q++){

        //Active set reset for it_warm iterations
        for (int m=0; m<it_warm; m++){
          if (lambda_it && screen_ind && a>0 && b>0){
            chng_flag = coord_des_onerun_wls(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                 family, delta, scr, inter, beta, mg, eta, resid, W, dev);
          } else{
            chng_flag = coord_des_onerun_wls(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                 family, delta, act, inter, beta, mg, eta, resid, W, dev);
          }

        }



        //Update active set
        int num_act = 0;
        int num_scr = 0;
        for (int j=0;j<pp;j++){
          if ((abs(beta[j])>0.0||(act_vec[j]>0.0))){ //
            act[j] = true;
            num_act ++;
            scr[j] = true;
            num_scr ++;
          }
          else{
            scr[j] = false;
            act[j] = false;
          }
        }

        //Update screen set
        if (lambda_it && screen_ind){
          if (a!= 0 && b!=0) {
            for (int g=0; g<J; g++) {
              cur_lambda = lambda*mg[g];
              for (int j=K1[g];j<K1[g+1];j++){

                if (scr[j]){

                  cj = wcrossprod(X, resid, W, nn, j)/((double)nn);
                  vj = wsqsum(X, W, nn, j)/((double)nn);

                  //int gind = floor((double)g/2.0); //index for sibling group

                  thresh = delta[g]+vj*gamma/(vj*gamma-delta[g]/cur_lambda)*mg[g]*(lambda-lambda_vec[a-1]);
                  if (abs(cj) < thresh || !scr[K1[g]]) {
                    scr[j] = false;
                    num_scr --;
                  }
                }
              }
            }
          }
        }


        //Cycle on active set
        it_inner = 0; //inner iteration count
        cont = true; //continue flag
        chng_flag = false; //change flag

        while (cont){
          //cout << "it_inner: " << it_inner << endl;

          //Increment count and update flags
          it_inner ++;
          if (lambda_it && screen_ind && a>0 && b>0){
            chng_flag = coord_des_onerun_wls(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                 family, delta, scr, inter, beta, mg, eta, resid, W, dev);
          } else {
            chng_flag = coord_des_onerun_wls(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                 family, delta, act, inter, beta, mg, eta, resid, W, dev);
          }

          //Update cont flag for termination
          if ( (it_inner >= it_max_reset)||(!chng_flag) ){
            cont = false;
          }
        }//end while


        //Rcout << accumulate(act.begin(),act.end(),0) << endl;
        //Update active set
        num_act = 0;
        num_scr = 0;
        for (int j=0;j<pp;j++){
          if ((abs(beta[j])>0.0)){ //||(act_vec[j]>0.0)
            act[j] = true;
            num_act ++;
            scr[j] = true;
            num_scr ++;
          }
          else{
            scr[j] = false;
            act[j] = false;
          }
        }

      }

      cycend:


        //Copy into beta_mat, and warm-start next cycle
        int betacount = 0;
      int betanz = 0;
      for (int k=0;k<pp;k++){
        if (abs(beta[k])>0.0){
          betanz++;
        }
        beta_mat(betacount,a) = beta[k];
        betacount++;
      }
      // for (int k=0;k<pcme;k++){
      //   if (abs(beta_cme[k])>0.0){
      //     betanz++;
      //   }
      //   beta_mat(betacount,a) = beta_cme[k];
      //   betacount++;
      // }
      nz(a,b) = betanz;
      inter_mat(a,b)= inter;
      dev_mat(a,b) = dev;

      //Copy deltas
      for (int k=0;k<J;k++){
        delta_mat(k,a) = delta[k];
      }

      //Copy residuals
      for (int k=0;k<nn;k++){
        resid_mat(k,a) = resid[k];
      }

      //Copy screening data
      for (int k=0;k<pp;k++){
        if (lambda_it && screen_ind && a>0 && b>0){
          scr_mat(k,a) = scr[k];
        } else {
          scr_mat(k,a) = act[k];
        }
      }
      // for (int k=0;k<pcme;k++){
      //   if (lambda_it && screen_ind && a>0 && b>0){
      //     scr_mat(pme+k,a) = scr_cme[k];
      //   } else {
      //     scr_mat(pme+k,a) = act_cme[k];
      //   }
      // }

    }//end nlambda.sib (a)

    //Copy into beta_cube
    beta_cube.slice(b) = beta_mat;
    delta_cube.slice(b) = delta_mat;
    resid_cube.slice(b) = resid_mat;
    scr_cube.slice(b) = scr_mat;

  }//end nlambda.cou (b)

  //Rescale betas and compute intercepts
  // arma::mat inter_mat(niter_1,niter_2);
  for (int b=0; b<niter_2; b++){ //iterate over cousins...
    for (int a=0; a<niter_1; a++){ //iterate over siblings...
      //Rescale betas to original scale
      inter_mat(a,b) = inter_mat(a,b);
      dev_mat(a,b) = dev_mat(a,b);
      for (int k=0;k<pp;k++){
        // beta_cube(k,a,b) = beta_cube(k,a,b);
        beta_cube(k,a,b) = beta_cube(k,a,b)/XX_sl(k);
      }
      // for (int k=0;k<pcme;k++){
      //   // beta_cube(pme+k,a,b) = beta_cube(pme+k,a,b);
      //   beta_cube(pme+k,a,b) = beta_cube(pme+k,a,b)/XX_cme_sl(k);
      // }
    }
  }

  return (List::create(Named("coefficients") = beta_cube,
                       Named("intercept") = inter_mat,
                       Named("residuals") = resid_cube,
                       Named("deviation") = dev_mat,
                       Named("nzero") = nz,
                       Named("lambda") = lambda_vec,
                       Named("act") = scr_cube,
                       Named("gamma") = gamma,
                       Named("tau") = tau
  ));

}


// [[Rcpp::export]]
List cme_gaussian(NumericMatrix& XX, NumericVector& yy,
                  NumericVector& K1,
                  NumericVector& lambda_vec,
                  NumericVector& gamma_vec, NumericVector& tau_vec,
                  NumericVector& XX_sl, NumericVector& beta_vec, NumericVector& act_vec, NumericVector& multiplier,
                  double lambda_max, int it_max, int it_warm, int reset, bool screen_ind) {
  // // [[Rcpp::plugins(openmp)]]
  //------------------------------------------------------------
  // XX - Full model matrix including both ME and CME effects (assume normalized)
  // yy - Response vector of length nn
  // family- family of GLM
  // lambda_sib_vec - Vector of sibling penalties (decr. sequence)
  // lambda_cou_vec - Vector of cousin penalties (decr. sequence)
  // tau - Exponential penalty parameter
  // gamma - MC+ non-convex penalty parameter
  // beta_vec - Initial beta value
  // it_max - Maximum iterations for coordinate descent
  //------------------------------------------------------------

  //Variable initialization
  int pp = XX.ncol(); //# of CMEs
  int nn = XX.nrow(); //# of observations
  int nlambda = lambda_vec.size();
  int it_inner = 0;
  int it_max_reset = it_max / reset;
  bool cont = true;
  bool chng_flag = false;
  //double mu = 0;
  int J = K1.size() - 1;
  int pme = J/2; //# of MEs

  // Extract the family type from CharacterVector

  //Vectorize model matrices
  vector<double> X(nn*pp); //for ME
  //vector<double> X_cme(nn*pcme); //for CME
  for (int i=0;i<pp;i++){
    for (int j=0;j<nn;j++){
      X[i*nn+j] = XX(j,i);
    }
  }

  //Check whether lambda is to be iterated or not
  bool lambda_it;
  int niter_1; //Number to iterate first
  int niter_2; //Number to iterate next
  if (gamma_vec.size()>1){ //Iterate on gamma and tau
    lambda_it = false;
    // niter_1 = gamma_vec.size(); //ch
    // niter_2 = tau_vec.size();
    niter_1 = tau_vec.size();
    niter_2 = gamma_vec.size();
  }
  else{
    lambda_it = true;
    niter_1 = nlambda;
    niter_2 = 1;
  }

  //Containers for beta and active set (alpha)
  arma::cube beta_cube(pp,niter_1,niter_2); //betas to return
  arma::cube delta_cube(J,niter_1,niter_2); //deltas to return
  arma::mat nz(niter_1,niter_2);
  arma::mat inter_mat(niter_1,niter_2); //intercept to return
  //arma::mat dev_mat(niter_1,niter_2); //deviation to return
  arma::mat beta_mat(pp,niter_1);
  arma::mat delta_mat(J,niter_1);

  vector<double> beta(pp,0.0); //for MEs
  for (int i=0;i<pp;i++){
    beta[i] = beta_vec[i];
  }

  //Set all factors as active to begin
  vector<bool> act(pp,true); //Current active set
  //vector<bool> act_cme(pcme,true);
  vector<bool> scr(pp,true); //Screened active set
  //vector<bool> scr_cme(pcme,true);
  //bool kkt_bool;


  //// Containers for linearized slopes Delta
  //vector<double> delta_sib(uniqueEffects.size(), 0.0); // Linearized penalty for siblings (sib(A), sib(B), ...)
  //vector<double> delta_cou(uniqueEffects.size(), 0.0); // Linearized penalty for cousins (cou(A), cou(B), ...)
  vector<double> mg(J,1);
  //vector<double> m_cme(pcme,1);
  for (int i=0;i<J;i++){
    mg[i] = multiplier[i];
  }
  // for (int i=0;i<pcme;i++){
  //   m_cme[i] = multiplier[pme+i];
  // }

  //Containers for linearized slopes Delta
  vector<double> delta(J); //Linearized penalty for siblings (sib(A), sib(B), ...)
  double lambda; //Current penalties
  double cur_delta; //Current delta vector
  double cur_lambda;
  double gamma;
  double tau;
  lambda = lambda_vec[0];
  gamma = gamma_vec[0];
  tau = tau_vec[0];

  //Update resid (eta)
  //vector<double> eta(nn);
  vector<double> resid(nn); //Residual vector
  arma::cube resid_cube(nn,niter_1,niter_2);
  arma::mat resid_mat(nn,niter_1);
  arma::cube scr_cube(pp,niter_1,niter_2); //screening vector
  arma::mat scr_mat(pp,niter_1);


  double inter= 0.0; //for intercept
  //double inprod = 0.0; //inner product
  double cj = 0.0;
  //double vj = 0.0;
  double thresh = 0.0; //threshold for screening
  //int size = 0;
  int num_act = 0;
  int num_scr = 0;

  double ymean = 0.0;
  for (int i=0;i<nn;i++){
    ymean += (1.0/(double)nn)*yy[i];
  }
  for (int i=0; i<nn; i++) {
    resid[i] = yy(i) - ymean;
  }


  // Optimize for each penalty combination
  // #pragma omp parallel for
  for (int b=0; b<niter_2; b++){ //iterate over cousins...

    for (int a=0; a<niter_1; a++){ //iterate over siblings...

      //cout << "Tuning ... a = " << a << ", b = " << b << endl;//Update iteration variables
      if (lambda_it){
        lambda = lambda_vec[a];
      }
      else{
        // gamma = gamma_vec[a]; //ch
        // tau = tau_vec[b];
        tau = tau_vec[a];
        gamma = gamma_vec[b];
      }

      //Return trivial solution of \beta=0 when \lambda_s + \lambda_c >= \lambda_max
      // if ( (lambda[0]+lambda[1]) >= lambda_max){
      if ( (a==0) || ( lambda >= lambda_max) ){
        for (int i=0;i<pp;i++){//reset beta
          beta[i] = 0.0;
        }
        // for (int i=0;i<pcme;i++){
        //   beta_cme[i] = 0.0;
        // }
        for (int i=0;i<nn;i++){//reset residuals
          resid[i] = yy(i) - ymean;
        }
        num_act = 0;
        num_scr = 0;
        for (int i=0;i<pp;i++){//reset active flag
          act[i] = true;
          scr[i] = true;
          num_act ++;
          num_scr ++;
        }
        //cout << "num_act: " << num_act << endl;
        if ( lambda >= lambda_max){
          goto cycend;
        }
      }

      // // RESET AFTER EACH RUN
      for (int i=0;i<pp;i++){//reset beta
        beta[i] = 0.0;
      }
      // for (int i=0;i<pcme;i++){
      //   beta_cme[i] = 0.0;
      // }
      for (int i=0;i<nn;i++){//reset residuals
        resid[i] = yy(i) - ymean;
      }

      //Recompute deltas
      fill(delta.begin(),delta.end(),lambda);
      for (int g=0; g<J; g++) {

        cur_lambda = lambda*mg[g];
         //assigns each element the value lambda

        for (int j=K1[g];j<K1[g+1];j++){
          //int gind = floor((double)g/2.0); //index for sibling or cousin group
          delta[g] = delta[g] * mg[g] * (exp(-(tau/cur_lambda) * mcp(beta[j],cur_lambda,gamma) )) ;
        }
      }

      //Coordinate descent with warm active set resets
      for (int q=0; q<reset; q++){

        //Active set reset for it_warm iterations
        for (int m=0; m<it_warm; m++){
          if (lambda_it && screen_ind && a>0 && b>0){
            chng_flag = coord_des_onerun_gaussian(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                  delta, scr, inter, beta, mg, resid);
          } else{
            chng_flag = coord_des_onerun_gaussian(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                  delta, act, inter, beta, mg, resid);
          }

        }



        //Update active set
        int num_act = 0;
        int num_scr = 0;
        for (int j=0;j<pp;j++){
          if ((abs(beta[j])>0.0||(act_vec[j]>0.0))){ //
            act[j] = true;
            num_act ++;
            scr[j] = true;
            num_scr ++;
          }
          else{
            scr[j] = false;
            act[j] = false;
          }
        }

        //cout << "warm act:" << num_act << endl;

        //Update screen set
        if (lambda_it && screen_ind){
          if (a!= 0 && b!=0) {
            for (int g=0; g<J; g++) {
              cur_lambda = lambda*mg[g];
              for (int j=K1[g];j<K1[g+1];j++){

                if (scr[j]){

                  cj = crossprod(X, resid, nn, j)/((double)nn);

                  //int gind = floor((double)g/2.0); //index for sibling group

                  thresh = delta[g]+gamma/(gamma-delta[g]/cur_lambda)*mg[g]*(lambda-lambda_vec[a-1]);
                  if (abs(cj) < thresh || !scr[K1[g]]) {
                    scr[j] = false;
                    num_scr --;
                  }
                }
              }
            }
          }
        }

        //cout << "strong set:" << num_scr << endl;


        //Cycle on active set
        it_inner = 0; //inner iteration count
        cont = true; //continue flag
        chng_flag = false; //change flag

        while (cont){
          //cout << "it_inner: " << it_inner << endl;

          //Increment count and update flags
          it_inner ++;
          if (lambda_it && screen_ind && a>0 && b>0){
            chng_flag = coord_des_onerun_gaussian(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                  delta, scr, inter, beta, mg, resid);
          } else {
            chng_flag = coord_des_onerun_gaussian(pme, nn, K1, lambda, cur_delta, chng_flag, tau, gamma, X, yy,
                                                  delta, act, inter, beta, mg, resid);
          }

          //Update cont flag for termination
          if ( (it_inner >= it_max_reset)||(!chng_flag) ){
            cont = false;
          }
        }//end while

        //cout << "it_inner: " << it_inner << endl;

        //Rcout << accumulate(act.begin(),act.end(),0) << endl;
        //Update active set
        num_act = 0;
        num_scr = 0;
        for (int j=0;j<pp;j++){
          if ((abs(beta[j])>0.0)){ //||(act_vec[j]>0.0)
            act[j] = true;
            num_act ++;
            scr[j] = true;
            num_scr ++;
          }
          else{
            scr[j] = false;
            act[j] = false;
          }
        }

        //cout << "converge act set" << num_act << endl;

      }

      cycend:


        //Copy into beta_mat, and warm-start next cycle
        int betacount = 0;
      int betanz = 0;
      for (int k=0;k<pp;k++){
        if (abs(beta[k])>0.0){
          betanz++;
        }
        beta_mat(betacount,a) = beta[k];
        betacount++;
      }

      nz(a,b) = betanz;

      //Copy deltas
      for (int k=0;k<J;k++){
        delta_mat(k,a) = delta[k];
      }

      //Copy residuals
      for (int k=0;k<nn;k++){
        resid_mat(k,a) = resid[k];
      }

      //Copy screening data
      for (int k=0;k<pp;k++){
        if (lambda_it && screen_ind && a>0 && b>0){
          scr_mat(k,a) = scr[k];
        } else {
          scr_mat(k,a) = act[k];
        }
      }

    }//end nlambda.sib (a)

    //Copy into beta_cube
    beta_cube.slice(b) = beta_mat;
    delta_cube.slice(b) = delta_mat;
    resid_cube.slice(b) = resid_mat;
    scr_cube.slice(b) = scr_mat;

  }//end nlambda.cou (b)

  //Rescale betas and compute intercepts
  //arma::mat inter_mat(niter_1,niter_2);
  vector<double> product(nn,0.0);
  for (int b=0; b<niter_2; b++){ //iterate over cousins...
    for (int a=0; a<niter_1; a++){ //iterate over siblings...
      //Rescale betas to original scale
      for (int k=0;k<pp;k++){
        // beta_cube(k,a,b) = beta_cube(k,a,b);
        beta_cube(k,a,b) = beta_cube(k,a,b)/XX_sl(k);
        for (int j = 0; j < nn; j++) {
          product[j] = X[k * nn + j] * beta_cube.slice(b).col(a)[k];
        }
      }
      for (int j = 0; j < nn; j++) {
        product[j] = yy[j] - product[j]; // Subtract from yy to get residuals
      }
      inter_mat(a,b) = sum(product,nn)/double(nn);
    }
  }

  return (List::create(Named("coefficients") = beta_cube,
                       Named("intercept") = inter_mat,
                       Named("residuals") = resid_cube,
                       Named("nzero") = nz,
                       Named("lambda") = lambda_vec,
                       Named("act") = scr_cube,
                       Named("gamma") = gamma,
                       Named("tau") = tau
  ));

}


