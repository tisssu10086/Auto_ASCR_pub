// #include <Rcpp.h>
// using namespace Rcpp;
#include <TMB.hpp>
#include <cmath>
// #include<iostream>


template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_MATRIX(capt_bin);
  DATA_MATRIX(capt_ss);
  DATA_MATRIX(capt_toa);
  
  DATA_MATRIX(masks);
  DATA_MATRIX(traps);
  DATA_SCALAR(survey_t);
  DATA_SCALAR(area);
  DATA_SCALAR(threshold);
  DATA_SCALAR(precision);
  
  PARAMETER(b0_link);
  PARAMETER(b1_link);
  PARAMETER(sigma_ss_link);
  PARAMETER(sigma_toa_link);
  Type b0 = b0_link;
  Type b1 = exp(b1_link);
  Type sigma_ss = exp(sigma_ss_link);
  Type sigma_toa = exp(sigma_toa_link);
  
  
  // get dimension of data
  int n_obs = capt_bin.rows();
  int n_masks = masks.rows();
  int n_traps = traps.rows();
  
  // get distances, signal strength, probability for each mask points
  matrix<Type> mask_dists(n_masks, n_traps);
  matrix<Type> mask_ss(n_masks, n_traps);
  matrix<Type> mask_probs(n_masks, n_traps);
  vector<Type> p_det(n_masks);
  p_det.setZero();
  p_det = p_det + 1;
  Type esa = Type(0);


  for (int i=0; i<n_masks; i++){
    for (int j=0; j<n_traps; j++){
      // get distance between masks and traps
      mask_dists(i,j) = sqrt(pow(masks(i,0) - traps(j,0), 2) + pow(masks(i,1)- traps(j,1), 2));
      // get signal strength for corresponding distance
      // if (mask_dists(i,j) > Type(1)){
      //   // mask_ss(i,j) = b0 -log10(mask_dists(i,j))* Type(20) - b1*(mask_dists(i,j) - Type(1));
      // }
      // else {
      //   mask_ss(i,j) = b0;
      // }
      mask_ss(i,j) = b0 - b1*(mask_dists(i,j));
      // get detection probs for corresponding signal strength using cdf of Normal dist
      mask_probs(i,j) = pnorm(mask_ss(i,j), threshold, sigma_ss);
      p_det[i] *= 1-mask_probs(i,j);
    }
    p_det[i] = 1 - p_det[i];
    esa += p_det[i];
  }

  matrix<Type> log_f_capt(n_obs, n_masks);
  log_f_capt.setZero();
  matrix<Type> log_f_sig(n_obs, n_masks);
  log_f_sig.setZero();
  matrix<Type> log_f_toa(n_obs, n_masks);
  vector<Type> f_mix(n_obs);
  f_mix.setZero();
  Type num_det;
  Type sum_toa_err;
  Type sum_squ_toa_err;
  Type nll = Type(0.0);
  // parallel_accumulator<Type> nll(this);

  for (int i=0; i<n_obs; i++){
    for (int j=0; j<n_masks; j++){
      // set number of detection to 0 as default
      num_det = Type(0);
      sum_toa_err = Type(0);
      sum_squ_toa_err = Type(0);
      for (int k=0; k<n_traps; k++){
        // likelihood if binary capt is 1, log_f_capt demilish with normalise of log_f_sig
        if (capt_bin(i,k) == Type(1)){
          log_f_sig(i,j) += dnorm(capt_ss(i,k), mask_ss(j,k), sigma_ss, int(1));
          // if binary capt is 0, add negative binominal probability
        } else {
          log_f_capt(i,j) += log(1 - mask_probs(j,k) + DBL_MIN);
        }
        // get sum toa error for toa likelihood usage
        if (capt_bin(i,k) == Type(1)){
          // get number of detection
          num_det += capt_bin(i,k);
          // get weighted toa error
          sum_toa_err += (capt_toa(i,k) - mask_dists(j,k)/Type(330));
          sum_squ_toa_err += pow(capt_toa(i,k) - mask_dists(j,k)/Type(330), 2);
        }
      }
      // toa error likelihood for det number more than once
      if (num_det > Type(1)){
        log_f_toa(i,j) = (sum_squ_toa_err -  pow(sum_toa_err, 2)/num_det)/(-Type(2)* pow(sigma_toa,2))
          + Type(0.5)* (Type(1)- num_det) *(log(Type(2)* M_PI) + Type(2)*log(sigma_toa)) - log(survey_t) - Type(0.5)* log(num_det);
      } else {
        log_f_toa(i,j) = Type(0);
      }
      // sum all masks into f_mix
      f_mix[i] += exp(log_f_capt(i,j) + log_f_sig(i,j)+ log_f_toa(i,j));
    }
    // normalise with f_loc density which is 1/esa
    nll -= log(f_mix[i] / esa + DBL_MIN);
  }
  
  Type D_c = n_obs * precision/(esa * survey_t * area);
  
  REPORT(D_c);
  REPORT(b0);
  REPORT(b1);
  REPORT(sigma_ss);
  REPORT(sigma_toa);
  REPORT(esa);

  return nll;
}
