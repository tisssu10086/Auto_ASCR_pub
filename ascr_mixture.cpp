// #include <Rcpp.h>
// using namespace Rcpp;
#include <TMB.hpp>
#include <cmath>
#include<iostream>

// This is model that model the confidence as observation with beta distribution, 
// and bin capt as latent variable
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_MATRIX(capt_bin);
  DATA_MATRIX(capt_ss);
  DATA_MATRIX(capt_toa);
  DATA_MATRIX(capt_conf);
  
  DATA_MATRIX(masks);
  DATA_MATRIX(traps);
  DATA_SCALAR(survey_t);
  DATA_SCALAR(area);
  DATA_SCALAR(threshold);
  
  PARAMETER(b0_link);
  PARAMETER(b1_link);
  PARAMETER(sigma_ss_link);
  PARAMETER(sigma_toa_link);
  Type b0 = b0_link;
  Type b1 = exp(b1_link);
  Type sigma_ss = exp(sigma_ss_link);
  Type sigma_toa = exp(sigma_toa_link);
  
  PARAMETER(noise_b0_link);
  PARAMETER(noise_b1_link);
  PARAMETER(noise_sigma_ss_link);
  Type noise_b0 = noise_b0_link;
  Type noise_b1 = exp(noise_b1_link);
  Type noise_sigma_ss = exp(noise_sigma_ss_link);

  
  // get the kernel density data 
  DATA_VECTOR(conf_tp);
  DATA_VECTOR(conf_fp);
  // and input the bandwidth
  DATA_SCALAR(bandwidth_tp);
  DATA_SCALAR(bandwidth_fp);
  int num_tp = conf_tp.size();
  int num_fp = conf_fp.size();
  
  
  PARAMETER(zeta_link);
  Type zeta = invlogit(zeta_link);

  
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
  
  //mesh calculation for target animal
  for (int i=0; i<n_masks; i++){
    for (int j=0; j<n_traps; j++){
      // get distance between masks and traps
      mask_dists(i,j) = sqrt(pow(masks(i,0) - traps(j,0), 2) + pow(masks(i,1)- traps(j,1), 2));
      // get signal strength for corresponding distance
      // if (mask_dists(i,j) > Type(1)){
      //   mask_ss(i,j) = b0 - b1*(mask_dists(i,j));
      // }
      // else {
      //   mask_ss(i,j) = b0;
      // }
      mask_ss(i,j) = b0 - b1*(mask_dists(i,j));
      // get detection probs for corresponding signal strength
      mask_probs(i,j) = pnorm(mask_ss(i,j), threshold, sigma_ss);
      p_det[i] *= 1-mask_probs(i,j);
    }
    p_det[i] = 1 - p_det[i];
    esa += p_det[i];
  }
  
  // get distances, signal strength, probability for each mask points for noise
  matrix<Type> noise_mask_ss(n_masks, n_traps);
  matrix<Type> noise_mask_probs(n_masks, n_traps);
  vector<Type> noise_p_det(n_masks);
  noise_p_det.setZero();
  noise_p_det = noise_p_det + 1;
  Type noise_esa = Type(0);
  
  //mesh calculation for noise
  for (int i=0; i<n_masks; i++){
    for (int j=0; j<n_traps; j++){
      // get signal strength for corresponding distance
      // if (mask_dists(i,j) > Type(1)){
      //   noise_mask_ss(i,j) = noise_b0 - noise_b1*(mask_dists(i,j));
      // }
      // else {
      //   noise_mask_ss(i,j) = noise_b0;
      // }
      noise_mask_ss(i,j) = noise_b0 - noise_b1*(mask_dists(i,j));
      // get detection probs for corresponding signal strength
      noise_mask_probs(i,j) = pnorm(noise_mask_ss(i,j), threshold, noise_sigma_ss);
      noise_p_det[i] *= 1- noise_mask_probs(i,j);
    }
    noise_p_det[i] = 1 - noise_p_det[i];
    noise_esa += noise_p_det[i];
  }
  
  
  
  // value for inner loop
  // value that can been initialized to 0 before lop
  Type log_f_capt;
  Type log_f_sig;
  Type log_f_toa;
  Type log_f_loc;

  
  //value need to be initialized within loop
  Type num_det;
  Type sum_toa_err;
  Type sum_squ_toa_err;
  
  // value for outer loop
  Type f_mix_tp;
  Type f_mix_fp;
  vector<Type> f_mix_latent(int(2));
  vector<Type> f_latent_conf(int(2));
  vector<Type> f_mix(n_obs);
  vector<Type> post_obs(n_obs);
  f_mix.setZero();
  post_obs.setZero();
  Type nll = Type(0.0);
  // Type kde;
  Type conf_sum;
  
  // loop for each observation
  for (int i=0; i<n_obs; i++){
    // pre allocate 0 for mix likelihood based on latent bin-capt
    f_mix_latent.setZero();
    // // pre allocate 1 for latent bin-capt likelihood based on conf
    f_latent_conf.setZero();
    // f_latent_conf += 1;
    
    // set number of detection to 0 as default
    num_det = Type(0);
    conf_sum = Type(0.0);
    // get likelihood for confidence with beta distribution
    for (int k=0; k<n_traps; k++){
      if (capt_bin(i, k) == Type(1)){
        // get number of detection
        num_det += capt_bin(i,k);
        conf_sum += capt_conf(i, k);
      }
    }
    // traverse all possible latent situation
    for (int m = 0; m < 2; m++){
      if (m == int(1)){
        for (int nval = 0; nval < num_tp; nval++){
          f_latent_conf[m] += dnorm(conf_sum/num_det, conf_tp[nval], bandwidth_tp, int(0)) / num_tp;
        }
      } else {
        for (int nval = 0; nval < num_fp; nval++){
          f_latent_conf[m] += dnorm(conf_sum/num_det, conf_fp[nval], bandwidth_fp, int(0)) / num_fp;
        }
      }
      
      // likelihood inside location integration
      for (int j=0; j<n_masks; j++){
        sum_toa_err = Type(0);
        sum_squ_toa_err = Type(0);
        for (int k=0; k<n_traps; k++){
          // get det number for each observation
          // get sum toa error for toa likelihood usage
          if (capt_bin(i,k) == Type(1)){
            // get weighted toa error
            sum_toa_err += (capt_toa(i,k) - mask_dists(j,k)/Type(330));
            sum_squ_toa_err += pow(capt_toa(i,k) - mask_dists(j,k)/Type(330), int(2));
          } 
        }
        
        // init log_likelihood
        log_f_capt = Type(0);
        log_f_sig = Type(0);
        log_f_toa = Type(0);
        // likelihood for det number more than zero
        if (m == int(1)){
          for (int k=0; k<n_traps; k++){
            // signal strength likelihood
            if (capt_bin(i,k) == Type(1)){
              log_f_sig += dnorm(capt_ss(i,k), mask_ss(j,k), sigma_ss, int(1));
            } else {
              log_f_capt += log(1 - mask_probs(j,k) + DBL_MIN);
            }
          } 
          // // location likelihood (can be add out of sum)
          // log_f_loc = -log(esa + DBL_MIN);
          // toa error likelihood for det number more than once, otherwise likelihood is set to 1
          if (num_det > Type(1)){
            log_f_toa = (sum_squ_toa_err -  pow(sum_toa_err, 2)/num_det)/(-Type(2)* pow(sigma_toa,2))
            + Type(0.5)* (Type(1)- num_det) *(log(Type(2)* M_PI) + Type(2)*log(sigma_toa)) - log(survey_t) - Type(0.5)* log(num_det);
          } 

        } else { // likelihood for fp
          for (int k=0; k<n_traps; k++){
            // signal strength likelihood
            if (capt_bin(i,k) == Type(1)){
              log_f_sig += dnorm(capt_ss(i,k), noise_mask_ss(j,k), noise_sigma_ss, int(1));
            } else{
              log_f_capt+= log(1 - noise_mask_probs(j,k) + DBL_MIN);
            }
          }
          // // location likelihood (can be add out of sum)
          // log_f_loc = -log(noise_esa + DBL_MIN);
          // toa error likelihood for det number more than once, otherwise likelihood is set to 1
          if (num_det > Type(1)){
            log_f_toa = (sum_squ_toa_err -  pow(sum_toa_err, 2)/num_det)/(-Type(2)* pow(sigma_toa,2))
            + Type(0.5)* (Type(1)- num_det) *(log(Type(2)* M_PI) + Type(2)*log(sigma_toa)) - log(survey_t) - Type(0.5)* log(num_det);
          }
        }
        // sum all masks into f_mix
        f_mix_latent[m] += exp(log_f_capt + log_f_sig + log_f_toa);
      }
    }
    
    // the first value (m = 0) means there is not detection
    f_mix_fp = (f_mix_latent[0] + DBL_MIN) * (f_latent_conf[0] + DBL_MIN) * (Type(1) - zeta) / (zeta * esa + (Type(1) - zeta)* noise_esa);
    // debug
    // f_mix_fp = (f_mix_latent[0] + DBL_MIN) * (Type(1) - zeta) / (zeta * esa + (Type(1) - zeta)* noise_esa);
    // m = 1 means positive
    f_mix_tp = (f_mix_latent[1] + DBL_MIN) * (f_latent_conf[1] + DBL_MIN) * zeta / (zeta * esa + (Type(1) - zeta)* noise_esa);
    // debug
     // f_mix_tp = (f_mix_latent[1] + DBL_MIN)  * zeta / (zeta * esa + (Type(1) - zeta)* noise_esa);
    
    // // add fp and tp likelihood as overall likelihood
    f_mix[i] = f_mix_tp +  f_mix_fp;
    

    // soft choice of whether observation as tp or fp, that is
    // using ratio of tp likelihood as ob value
    post_obs[i] = f_mix_tp / f_mix[i];
    
    // add f_mix for each observation to overall neg-log likelihood 
    nll -= log(f_mix[i] + DBL_MIN);
  }
  
  Type obs_tp = sum(post_obs);
  Type obs_fp = n_obs - obs_tp;
  Type D_tp = obs_tp/(esa * survey_t * area);
  Type D_fp = obs_fp/(noise_esa * survey_t * area);
  
  
  REPORT(D_tp);
  REPORT(obs_tp);
  REPORT(b0);
  REPORT(b1);
  REPORT(sigma_ss);
  REPORT(sigma_toa);
  REPORT(esa);
  REPORT(D_fp);
  REPORT(obs_fp);
  REPORT(noise_b0);
  REPORT(noise_b1);
  REPORT(noise_sigma_ss);
  REPORT(noise_esa);
  REPORT(zeta);
  
  return nll;
}
