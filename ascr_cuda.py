import torch 
import numpy as np
import torch.nn.functional as F



class ASCR_random_mix_kde():
    def __init__(self, capthist_data: dict, param_init: dict, device: str = 'cpu'):
        self.device = device
        self.lam = torch.tensor([0.368], dtype = torch.float64).to(device)
        self.pi = torch.tensor([np.pi], dtype = torch.float64).to(device)

        self.traps = torch.tensor(capthist_data['traps'], dtype = torch.float64).to(device)
        self.loc = torch.tensor(capthist_data['masks'], dtype = torch.float64).to(device)
        self.bin_capt = torch.tensor(capthist_data['capt_bin'], dtype = torch.float64).to(device)
        self.sig_capt = torch.tensor(capthist_data['capt_ss'], dtype = torch.float64).to(device)
        self.toa_capt = torch.tensor(capthist_data['capt_toa'], dtype = torch.float64).to(device)
        self.conf_capt = torch.tensor(capthist_data['capt_conf'], dtype = torch.float64).to(device)
        self.survey_t = torch.tensor(capthist_data['survey_t'], dtype = torch.float64).to(device)
        self.area = torch.tensor(capthist_data['area'], dtype = torch.float64).to(device)

        self.conf_tp = torch.tensor(capthist_data['param_val']['conf_tp'], dtype = torch.float64).to(device)
        self.conf_fp = torch.tensor(capthist_data['param_val']['conf_fp'], dtype = torch.float64).to(device)
        self.band_tp = torch.tensor(capthist_data['param_val']['band_tp'], dtype = torch.float64).to(device)
        self.band_fp = torch.tensor(capthist_data['param_val']['band_fp'], dtype = torch.float64).to(device)
        self.num_tp = self.conf_tp.shape[0]
        self.num_fp = self.conf_fp.shape[0]
        self.conf_tp = self.conf_tp.view(-1, 1, 1)
        self.conf_fp = self.conf_fp.view(-1, 1, 1)

        self.num_fp = torch.tensor(self.num_fp, dtype = torch.float64).to(device)
        self.num_tp = torch.tensor(self.num_tp, dtype = torch.float64).to(device)


        self.b0_link = torch.tensor(param_init['b0'], dtype = torch.float64).to(device).requires_grad_(True)
        self.b1_link = torch.log(torch.tensor(param_init['b1'], dtype = torch.float64)).to(device).requires_grad_(True)
        self.sigma_ss_link = torch.log(torch.tensor(param_init['sigma_ss'], dtype = torch.float64)).to(device).requires_grad_(True)
        self.sigma_t_link = torch.log(torch.tensor(param_init['sigma_t'], dtype = torch.float64)).to(device).requires_grad_(True)
        self.rss_coef0_link = torch.tensor(param_init['rss_coef0'], dtype = torch.float64).to(device).requires_grad_(True)
        self.rss_coef1_link = torch.log(torch.tensor(param_init['rss_coef1'], dtype = torch.float64)).to(device).requires_grad_(True)
        self.noise_b0_link = torch.tensor(param_init['noise_b0'], dtype = torch.float64).to(device).requires_grad_(True)
        self.noise_b1_link = torch.log(torch.tensor(param_init['noise_b1'], dtype = torch.float64)).to(device).requires_grad_(True)
        self.noise_sigma_ss_link = torch.log(torch.tensor(param_init['noise_sigma_ss'], dtype = torch.float64)).to(device).requires_grad_(True)
        self.zeta_link = torch.tensor(np.log(param_init['zeta']/(1 - param_init['zeta'])), dtype = torch.float64).to(device).requires_grad_(True)

        self.param_opt = {}    
        for param_name in param_init:
            self.param_opt[param_name] = param_init[param_name]
        self.param_opt['D_tp'] = None
        self.param_opt['D_fp'] = None
        self.param_opt['esa_tp'] = None
        self.param_opt['esa_fp'] = None
        self.param_opt['obs_tp'] = None
        self.param_opt['obs_fp'] = None

        self.nll = None
        self.esa = None
        self.esa_fp = None
        self.obs_tp = None
        self.obs_fp = None

    @staticmethod
    def cal_dis(loc, traps):
        return torch.cdist(loc, traps, p = 2)

    @staticmethod
    def cal_ss(d, b0, b1):
        zero = torch.zeros_like(d)
        one = torch.ones_like(d)   
        d_condi = torch.where(d > 1, one, zero)
        ss = b0 - (20 * torch.log10(d) + b1 * (d - 1)) * d_condi
        return(ss)
            
    def cal_log_prob(self, mu_ss, sigma_ss, rss_coef0, rss_coef1):
        mu = rss_coef1 * mu_ss + rss_coef0
        sigma = sigma_ss * rss_coef1
        cali_value = mu / torch.sqrt(1 + torch.pow(sigma, 2) * self.lam)
        log_prob = F.logsigmoid(cali_value)
        return log_prob

    @staticmethod
    def cal_prob_once(prob):
        p_avoid = torch.prod(1-prob, dim = 1)
        return 1 - p_avoid

    @staticmethod
    def cal_esa(prob_once):
        return torch.sum(prob_once)

    @staticmethod
    def cal_ll_det(prob, bin_capt):
        log_pbinom = torch.distributions.Binomial(probs= prob).log_prob(bin_capt.unsqueeze(1).expand(-1, prob.shape[0], -1))
        return torch.sum(log_pbinom, dim = 2) ## dim = (obs * mask * traps) -> (obs * mask)


    @staticmethod
    def cal_ll_sig(mu_ss, sigma_ss, rss_coef0, rss_coef1, sig_capt, bin_capt, log_prob):
        log_plogis = F.logsigmoid(rss_coef1 * sig_capt + rss_coef0).unsqueeze(1).expand(-1, log_prob.shape[0], -1)
        log_dnorm = torch.distributions.Normal(mu_ss, sigma_ss).log_prob(sig_capt.unsqueeze(1).expand(-1, log_prob.shape[0], -1))
        log_prob_expand = log_prob.unsqueeze(0).expand(log_plogis.shape[0], -1, -1)
        log_likeli_sig = (log_plogis + log_dnorm - log_prob_expand)* bin_capt.unsqueeze(1).expand(-1, log_prob.shape[0], -1)
        return torch.sum(log_likeli_sig, dim = 2)


    def cal_ll_toa(self, d, sigma_t, survey_t, toa_capt, bin_capt):
        emit_t = (toa_capt.unsqueeze(1).expand(-1, d.shape[0], -1) - d.unsqueeze(0).expand(toa_capt.shape[0], -1, -1)/330) *\
            bin_capt.unsqueeze(1).expand(-1, d.shape[0], -1)
        sum_emit_t = torch.sum(emit_t, dim = 2) 
        sum_squr_emit_t = torch.sum(torch.pow(emit_t, 2), dim = 2)
        num_det =  torch.sum(bin_capt.unsqueeze(1).expand(-1, d.shape[0], -1), dim = 2)
        log_likeli_toa = (sum_squr_emit_t - torch.pow(sum_emit_t, 2) / num_det) /\
            (-2 * torch.pow(sigma_t, 2)) + 0.5*(1- num_det) * (torch.log(2 * self.pi) +2 * torch.log(sigma_t)) -\
                torch.log(survey_t) - 0.5 * torch.log(num_det)
        log_likeli_toa = torch.where(num_det > 1, log_likeli_toa, torch.zeros_like(log_likeli_toa))
        return log_likeli_toa

    @staticmethod
    # def cal_likeli_conf(conf_tp, conf_fp, band_tp, band_fp, num_tp, num_fp, conf_capt, bin_capt):
    #     conf_capt = torch.where(bin_capt == 1, conf_capt, torch.ones_like(conf_capt))
    #     kde_fp = torch.distributions.Normal(conf_fp, band_fp)
    #     kde_tp = torch.distributions.Normal(conf_tp, band_tp)
    #     likeli_conf_neg_mat = torch.sum(torch.exp(kde_fp.log_prob(conf_capt)), dim = 0) / num_fp
    #     likeli_conf_pos_mat = torch.sum(torch.exp(kde_tp.log_prob(conf_capt)), dim = 0) / num_tp
    #     likeli_conf_neg = torch.prod(torch.where(bin_capt == 1, likeli_conf_neg_mat, torch.ones_like(conf_capt)), dim = 1)
    #     likeli_conf_pos = torch.prod(torch.where(bin_capt == 1, likeli_conf_pos_mat, torch.ones_like(conf_capt)), dim = 1)
    #     likeli_conf = torch.stack([likeli_conf_neg, likeli_conf_pos], dim = 1)
    #     return likeli_conf
    def cal_likeli_conf(conf_tp, conf_fp, band_tp, band_fp, num_tp, num_fp, conf_capt, bin_capt):
        conf_capt = torch.where(bin_capt == 1, conf_capt, torch.zeros_like(conf_capt))
        kde_fp = torch.distributions.Normal(conf_fp, band_fp)
        kde_tp = torch.distributions.Normal(conf_tp, band_tp)
        mean_conf_capt = torch.sum(conf_capt, dim = 1) / torch.sum(bin_capt, dim = 1)
        # print(mean_conf_capt.shape)
        likeli_conf_neg = torch.squeeze( torch.logsumexp(kde_fp.log_prob(mean_conf_capt), dim = 0), 0) - torch.log(num_fp)
        likeli_conf_pos = torch.squeeze( torch.logsumexp(kde_tp.log_prob(mean_conf_capt), dim = 0), 0) - torch.log(num_tp)
        # print(likeli_conf_neg.shape)

        # likeli_conf_neg_mat = torch.sum(torch.exp(kde_fp.log_prob(conf_capt)), dim = 0) / num_fp
        # print(likeli_conf_neg_mat.shape)
        # # likeli_conf_pos_mat = torch.sum(torch.exp(kde_tp.log_prob(conf_capt)), dim = 0) / num_tp
        # likeli_conf_neg = torch.prod(torch.where(bin_capt == 1, likeli_conf_neg_mat, torch.ones_like(conf_capt)), dim = 1)
        # print(likeli_conf_neg.shape)
        # # likeli_conf_pos = torch.prod(torch.where(bin_capt == 1, likeli_conf_pos_mat, torch.ones_like(conf_capt)), dim = 1)


        likeli_conf = torch.stack([likeli_conf_neg, likeli_conf_pos], dim = 1)
        # print(likeli_conf.shape)
        return likeli_conf
    
    @staticmethod
    def cal_likeli_cr(ll_det, ll_sig, ll_toa, noise_ll_det, noise_ll_sig, noise_ll_toa):
        ll_all = torch.logsumexp(ll_det + ll_sig + ll_toa, dim = 1)
        noise_ll_all = torch.logsumexp(noise_ll_det + noise_ll_sig + noise_ll_toa, dim = 1)
        likeli_cr = torch.stack([noise_ll_all, ll_all], dim = 1)
        return likeli_cr



    def ascr_func(self):
        ## change the space of parameters
        rss_coef0 = self.rss_coef0_link
        rss_coef1 = torch.exp(self.rss_coef1_link)
        b0 = self.b0_link   
        b1 = torch.exp(self.b1_link)
        sigma_ss = torch.exp(self.sigma_ss_link)
        sigma_t = torch.exp(self.sigma_t_link)
        noise_b0 = self.noise_b0_link
        noise_b1 = torch.exp(self.noise_b1_link)
        noise_sigma_ss = torch.exp(self.noise_sigma_ss_link)
        zeta = torch.sigmoid(self.zeta_link)
        ## calculate the likelihood
        d = self.cal_dis(self.loc, self.traps)
        ## calculate the likelihood for target signal
        mu_ss = self.cal_ss(d, b0, b1)
        log_prob = self.cal_log_prob(mu_ss, sigma_ss, rss_coef0, rss_coef1)
        prob = torch.exp(log_prob)
        prob_once = self.cal_prob_once(prob)
        esa = self.cal_esa(prob_once)
        ll_det = self.cal_ll_det(prob, self.bin_capt)
        ll_sig = self.cal_ll_sig(mu_ss, sigma_ss, rss_coef0, rss_coef1, self.sig_capt, self.bin_capt, log_prob)
        ll_toa = self.cal_ll_toa(d, sigma_t, self.survey_t, self.toa_capt, self.bin_capt)
        ## calculate the likelihood for noise signal
        noise_mu_ss = self.cal_ss(d, noise_b0, noise_b1)
        noise_log_prob = self.cal_log_prob(noise_mu_ss, noise_sigma_ss, rss_coef0, rss_coef1)
        noise_prob = torch.exp(noise_log_prob)
        noise_prob_once = self.cal_prob_once(noise_prob)
        noise_esa = self.cal_esa(noise_prob_once)
        noise_ll_det = self.cal_ll_det(noise_prob, self.bin_capt)
        noise_ll_sig = self.cal_ll_sig(noise_mu_ss, noise_sigma_ss, rss_coef0, rss_coef1, self.sig_capt, self.bin_capt, noise_log_prob)
        noise_ll_toa = ll_toa
        # calculate overall likelihood
        likeli_cr = self.cal_likeli_cr(ll_det, ll_sig, ll_toa, noise_ll_det, noise_ll_sig, noise_ll_toa)
        likeli_conf = self.cal_likeli_conf(self.conf_tp, self.conf_fp, self.band_tp, self.band_fp, self.num_tp, self.num_fp, self.conf_capt, self.bin_capt)
        esa_combine = esa * zeta + noise_esa * (1 - zeta)
        liekli_zeta = torch.stack([(1 - zeta)/ esa_combine , zeta / esa_combine], dim = 0)



        # log_likeli_neg_pos = likeli_cr + torch.log(likeli_conf) + torch.log(liekli_zeta)
        log_likeli_neg_pos = likeli_cr + likeli_conf + torch.log(liekli_zeta)



        ## use logsumexp to avoid underflow
        log_likeli_all = torch.logsumexp(log_likeli_neg_pos, dim = 1)
        ## calculate the negative log likelihood
        nll = -torch.sum(log_likeli_all, dim = 0)
        ## calculate identity posterior 
        likeli_neg_pos = torch.exp(log_likeli_neg_pos)
        likeli_all = torch.sum(likeli_neg_pos, dim = 1)
        ## save esa and nll, call n_obs_cali
        self.obs_tp = torch.sum(likeli_neg_pos[:,1]/likeli_all, dim = 0)
        self.obs_fp = torch.sum(likeli_neg_pos[:,0]/likeli_all, dim = 0)
        self.esa = esa
        self.esa_fp = noise_esa
        self.nll = nll
        return nll

    def optimize(self, max_iter = 150, lr = 1):
        params = [self.rss_coef0_link, self.rss_coef1_link, self.b0_link, self.b1_link, self.sigma_ss_link, self.sigma_t_link, 
        self.noise_b0_link, self.noise_b1_link, self.noise_sigma_ss_link, self.zeta_link]
        opt = torch.optim.LBFGS(params, lr = lr, max_iter = max_iter, max_eval = max_iter, tolerance_grad = 1e-05, tolerance_change = 1e-10, history_size = 100, line_search_fn = 'strong_wolfe')
        for _ in range(max_iter):
            def closure():
                opt.zero_grad()
                nll = self.ascr_func()
                nll.backward()
                return nll
            opt.step(closure)


    def output_param_opt(self):
        self.param_opt['rss_coef0'] = self.rss_coef0_link
        self.param_opt['rss_coef1'] = torch.exp(self.rss_coef1_link)
        self.param_opt['b0'] = self.b0_link
        self.param_opt['b1'] = torch.exp(self.b1_link)
        self.param_opt['sigma_ss'] = torch.exp(self.sigma_ss_link)
        self.param_opt['sigma_t'] = torch.exp(self.sigma_t_link)
        self.param_opt['noise_b0'] = self.noise_b0_link
        self.param_opt['noise_b1'] = torch.exp(self.noise_b1_link)
        self.param_opt['noise_sigma_ss'] = torch.exp(self.noise_sigma_ss_link)
        self.param_opt['zeta'] = torch.sigmoid(self.zeta_link)
        self.param_opt['D_tp'] = self.obs_tp / (self.esa * self.area[0] * self.survey_t[0])
        self.param_opt['D_fp'] = self.obs_fp / (self.esa_fp * self.area[0] * self.survey_t[0])
        self.param_opt['esa_tp'] = self.esa
        self.param_opt['esa_fp'] = self.esa_fp
        self.param_opt['obs_tp'] = self.obs_tp
        self.param_opt['obs_fp'] = self.obs_fp
        return self.param_opt


    def __call__(self):
        self.optimize()
        return self.output_param_opt()







