import torch 
import json
import numpy as np
import time
import torch.multiprocessing as mp
from pathlib import Path
import re

# from ascr_cuda import ASCR_conf
from ascr_cuda import ASCR_random_mix_kde





def main():
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # init parameter 
    param_init = {'b0': 0, 'b1': 0.1, 'sigma_ss': 5, 'sigma_t': 5, 'rss_coef0': 10, 'rss_coef1': 0.1, 
    'noise_b0': -30, 'noise_b1': 0.1, 'noise_sigma_ss': 15, 'zeta': 0.7}
    with open('gibbon_capthist.json') as f:
        capthist = json.load(f)
    # set param_init rss_coef
    param_init['rss_coef0'] = capthist['param_val']['rsscoef'][0]
    param_init['rss_coef1'] = capthist['param_val']['rsscoef'][1]
    ## run
    start = time.time()
    ascr_rmg = ASCR_random_mix_kde(capthist, param_init, device)
    nll_test = ascr_rmg.ascr_func()
    ascr_rmg.optimize(lr = 1)
    result = ascr_rmg.output_param_opt()
    end = time.time()
    ## change result to numpy array
    for key in result.keys():
        result[key] = result[key].cpu().detach().numpy()
    print(result)
    print('time cost: ', end - start)



if __name__ == '__main__':
    main()
