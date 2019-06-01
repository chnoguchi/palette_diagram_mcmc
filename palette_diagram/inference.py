import sys

sys.path.append('newman2016/')
from mcmc_newman2016 import *

def inference(cfg):

    cfg.algorithm_option = 3
    cfg.method = 'mcmc_nr'
    print('Performing MCMC_NR...')
    exec_mcmc(cfg.gml_filename, cfg.sample_interval, cfg.n_samples, cfg.relaxation_time, cfg.K)
    grp_assing_filename='newman2016/outputs/grp_assign.txt'
         
    print('Finish')
    return grp_assing_filename