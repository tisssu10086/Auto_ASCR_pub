library(ascr)
library(tidyverse)
library(ggplot2)
library(TMB)
require(EnvStats)
library(MASS)
library(mclust)
library(rlist)
library(foreach)
require(doParallel)
library(sn)
library(jsonlite)


## function to transfer table fp, tp to fp, tp vector
tb2paramval <- function(df){
  ## get df tp
  df.tp <- df %>%
    filter(!grepl("fp$", Label_index)) %>%
    filter(!grepl("^fp", Label_index))
  ## get df fp
  df.fp <- df %>%
    filter(!grepl("^tp", Label_index))
  
  ## estimate precision rate
  obs_fp <- nrow(df.fp)
  obs_tp <- nrow(df.tp)
  precision <- obs_tp / (obs_tp + obs_fp)
  
  ## get fp and tp confidence value 
  fp_columns <- grep("^Coef", names(df.fp), value = TRUE)
  ## change fp value to mean 
  fp_value <- c(apply(df.fp[fp_columns], 1, mean, na.rm = T))
  tp_columns <- grep("^Coef", names(df.tp), value = T)
  tp_value <- c(apply(df.tp[tp_columns], 1, mean, na.rm = T))
  
  ## cal standard deviation
  fp_sd <- unlist(df.fp[fp_columns] - fp_value)
  fp_sd <- fp_sd[!is.na(fp_sd)]
  hist(fp_sd[fp_sd != 0], breaks = 30)
  fp_sd <- sd(fp_sd[fp_sd != 0])
  
  tp_sd <- unlist(df.tp[tp_columns] - tp_value)
  tp_sd <- tp_sd[!is.na(tp_sd)]
  hist(tp_sd[tp_sd != 0], breaks = 30)
  tp_sd <- sd(tp_sd[tp_sd != 0])
  
  ## cal bandwidth
  band_tp <- bw.nrd0(tp_value)
  band_fp <- bw.nrd0(fp_value)
  
  conf_value <- data.frame(values = c(tp_value, fp_value), set = factor(c(rep(1, length(tp_value)), rep(0, length(fp_value)))))
  param_val = list(precision = precision, conf_set = conf_value, conf_fp = fp_value, conf_tp = tp_value, 
                   band_fp = band_fp, band_tp = band_tp, tp_sd = tp_sd, fp_sd = fp_sd, df = df)
  return (param_val)
}


## function to generate capture history from merged table
tb2capt <- function(traps, masks, cutoff, df, survey_t){
  ss_columns <- grep("^SPL", names(df), value = TRUE)
  capt_ss <- as.matrix(df[ss_columns])
  toa_columns <- grep("^Time", names(df), value = TRUE)
  capt_toa <- as.matrix(df[toa_columns])
  conf_columns <- grep("^Coef", names(df), value = TRUE)
  capt_conf <- as.matrix(df[conf_columns])
  # capt_conf <- logit(capt_conf)
  
  ## get rid of detection below -63
  capt_ss[capt_ss < cutoff] <- NA
  ## generate binary capt
  capt_bin <- capt_ss
  capt_bin[!is.na(capt_bin)] = 1
  capt_bin[is.na(capt_bin)] = 0
  ## remove capt_bin with all row as 0
  capt_ss <- capt_ss[apply(capt_bin, 1, function(x) !all(x == 0)),]
  capt_toa <- capt_toa[apply(capt_bin, 1, function(x) !all(x == 0)),]
  capt_conf <- capt_conf[apply(capt_bin, 1, function(x) !all(x == 0)),]
  capt_bin <- capt_bin[apply(capt_bin, 1, function(x) !all(x == 0)),]
  
  
  capt_ml = list(capt_bin = capt_bin, capt_ss = capt_ss, capt_toa = capt_toa, capt_conf = capt_conf,
                 masks = masks, traps = traps, survey_t = survey_t, area = attr(masks, 'area'),
                 threshold = cutoff)
  return(capt_ml)
}


## single ascr procedure
single_ascr <- function(param_init, param_val, ascr_model, capt_df){
  ## compile model if not compiled before
  if (!file.exists(paste0(ascr_model, '.so'))){
    compile(paste0(ascr_model, '.cpp'))
  }
  ## add parameter from validation set
  capt_df$precision <- param_val$precision
  capt_df$conf_tp <- param_val$conf_tp
  capt_df$conf_fp <- param_val$conf_fp
  capt_df$bandwidth_tp <- param_val$band_tp
  capt_df$bandwidth_fp <- param_val$band_fp
  ## inference
  dyn.load(dynlib(ascr_model))
  ## refit model if not converge
  convergency <- NaN
  cov_tol <- 1e-10
  while(convergency != 'relative convergence (4)' & cov_tol <= 1e-4){
    model <- MakeADFun(data = capt_df, parameters = param_init, DLL = ascr_model, silent = F)
    # config(tape.parallel = FALSE) have bug when there are multiple cpp file,
    # used for memory save, may not useful
    try(fit <-nlminb(model$par, model$fn, model$gr, control = list(rel.tol = cov_tol)))
    try(convergency <- fit$message)
    cov_tol <- cov_tol * 100
    print(model$report())
    param_elem <- unlist(model$report())
    ## free memory
    FreeADFun(model)
    gc()
  }
  ## if not converged, return NaN
  if (convergency != 'relative convergence (4)'){
    param_elem = param_elem + NaN
  }
  # change list to vector
  return(param_elem)
}





##############################################################
## running example for frog data 
# read table for survey and validation data
config_set <- lightfooti
survey_frog <- read.csv('frog_survey.csv')
survey_frog_tp <- survey_frog %>%
  filter(!grepl("fp$", Label_index)) %>%
  filter(!grepl("^fp", Label_index))
val_frog <- read.csv('frog_validation.csv')
## generate capthist for test set with only tp
capt_tp <- tb2capt(config_set$traps, create.mask(config_set$traps, buffer = 90, nx = 80, ny = 80), -65, survey_frog_tp, 30)
## generate capthist for test set with all data
capt_survey <- tb2capt(config_set$traps, create.mask(config_set$traps, buffer = 90, nx = 80, ny = 80), -65, survey_frog, 30)
## do ASCR without fp  
param_val <- list(precision = 1)
param_init <- list(b0_link = -20, b1_link = log(5), sigma_ss_link = log(5), sigma_toa_link = log(0.1))
tp_estimate <- single_ascr(param_init, param_val, 'ascr_canonical', capt_tp)
print(tp_estimate)
## do ASCR using canonical method
param_val <- tb2paramval(val_frog)
param_init <- list(b0_link = -20, b1_link = log(5), sigma_ss_link = log(5), sigma_toa_link = log(0.1))
canoni_estimate <- single_ascr(param_init, param_val, 'ascr_canonical', capt_survey)
print(canoni_estimate)
## do ASCR with mixture model
param_val <- tb2paramval(val_frog)
param_init <- list(b0_link = -20, b1_link = log(5), sigma_ss_link = log(5), sigma_toa_link = log(0.1), zeta_link = 1,
                   noise_b0_link = -60, noise_b1_link = log(5), noise_sigma_ss_link = log(5))
mixture_estimate <- single_ascr(param_init, param_val, 'ascr_mixture', capt_survey)
print(mixture_estimate)

###############################################################################
# running example for gibbon simulation 
gibbon_capt <- fromJSON('gibbon_capthist.json') 
param_init <- list(b0_link = 0, b1_link = log(0.1), sigma_ss_link = log(5), sigma_toa_link = log(5),
                   rss_coef_link = c(10, log(0.1)), noise_b0_link = -30, noise_b1_link = log(0.1),
                   noise_sigma_ss_link = log(15), zeta_link = 1)
param_val <- gibbon_capt$param_val
auto_mixture_estimate <- single_ascr(param_init, param_val, 'auto_mixture', gibbon_capt)
print(auto_mixture_estimate)
