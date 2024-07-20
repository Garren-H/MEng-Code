import numpy as np # type: ignore
import json
import os

# change stan tmpdir to home. Just a measure added for computations on the HPC which does not 
# like writing to /tmp
old_tmp = os.environ['TMPDIR'] # save previous tmpdir
os.environ['TMPDIR'] = '/home/22796002' # update tmpdir

import cmdstanpy # type: ignore

os.environ['TMPDIR'] = old_tmp # change back to old_tmp

import sys
import pandas as pd # type: ignore
from multiprocessing import Pool

# Append path to obtain other functions
sys.path.append('/home/22796002')

from generate_stan_model_code import generate_stan_code # type: ignore
from All_code import subsets
import k_means

# get arguments from command line
include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
variance_known = bool(int(sys.argv[2])) # True if you want to use known variance information
variance_MC_known = bool(int(sys.argv[3])) # True if you want to use known variance information for MC
func_groups_string = sys.argv[4] # functional groups to extract
functional_groups = func_groups_string.split(',') # restructure to numpy array 
chain_id = int(sys.argv[5]) # chain id


# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'

data_file = f'{path}/data.json'

path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}/Variance_MC_known_{variance_MC_known}'

print('Evaluating the following conditions for the Pure RK PMF Model:')
print(f'Include clusters: {include_clusters}')
print(f'Variance known: {variance_known}')
print(f'Variance MC known: {variance_MC_known}')

# compile stan model
exe_file = f'/home/22796002/Pure RK PMF/Stan Models/Pure_PMF_Include_clusters_{include_clusters}_Variance_known_{variance_known}_Variance_MC_known_{variance_MC_known}'
model = cmdstanpy.CmdStanModel(exe_file=exe_file, cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 1
threads_per_chain = 4

os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))


output_dir1 = f'{path}/Initializations/{chain_id}'
output_dir2 = f'{path}/MAP/{chain_id}'
inits2 = f'{output_dir2}/inits.json'
num_warmup = 1000
num_samples = 100

e=True
max_iter=20
iter=0
while e and iter<max_iter:
    fit = model.sample(data=data_file, output_dir=output_dir1,
                        refresh=100, iter_warmup=num_warmup, 
                        iter_sampling=num_samples, chains=chains, parallel_chains=chains, 
                        threads_per_chain=threads_per_chain, max_treedepth=5, inits=inits2)
    #save inits from previous step
    max_lp = np.argmax(fit.method_variables()['lp__'].T.flatten())
    dict_keys = list(fit.stan_variables().keys())
    init = {}
    for key in dict_keys:
        try:
            init[key] = fit.stan_variables()[key][max_lp].tolist()
        except:
            init[key] = fit.stan_variables()[key][max_lp]
    with open(inits2, 'w') as f:
        json.dump(init, f)
    del fit, init

    try:
        prev_files = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if not f.endswith('.json')]
        MAP = model.optimize(data=data_file, output_dir=output_dir2, inits=inits2, show_console=True,  iter=1000000, refresh=1000, 
                        algorithm='lbfgs', jacobian=False, tol_rel_grad=1e-20, tol_rel_obj=1e-20)
        e=False
        for f in prev_files:
            os.remove(f)
    except:
        delete_files = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if not f.endswith('.json')]
        for f in delete_files:
            os.remove(f)
        e=True
        iter+=1