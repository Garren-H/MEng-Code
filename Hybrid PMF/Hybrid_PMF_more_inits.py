import numpy as np # type: ignore
import json
import os

# change stan tmpdir to home. Just a measure added for computations on the HPC which does not 
# like writing to /tmp. My change to something else if ran on a different server where /home is limited
old_tmp = os.environ['TMPDIR'] # save previous tmpdir
os.environ['TMPDIR'] = '/home/ghermanus/lustre' # update tmpdir

import cmdstanpy # type: ignore

os.environ['TMPDIR'] = old_tmp # change back to old_tmp

import sys
import pandas as pd # type: ignore
from multiprocessing import Pool

sys.path.insert(0, '/home/ghermanus/lustre') # include home directory in path to call a python file

# import local function files

from All_code import subsets # python script that extracts data for functional group of interest
import k_means
from generate_stan_model_code import generate_stan_code #type: ignore

# get arguments from command line
include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
variance_known = bool(int(sys.argv[2])) # True if you want to use known variance information
func_groups_string = sys.argv[3] # functional groups to extract
functional_groups = func_groups_string.split('.') # restructure to numpy array 
chain_id = int(sys.argv[4]) # chain id

print('Evaluating the following conditions for the Hybrid Model:')
print(f'Include clusters: {include_clusters}')
print(f'Variance known: {variance_known}')
print('\n')

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'
data_file = f'{path}/data.json' # file where data is stored
path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}'

# set number of chains and threads per chain
chains = 1
threads_per_chain = 4
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Obtain stan model
model = cmdstanpy.CmdStanModel(exe_file=f'/home/ghermanus/lustre/Hybrid PMF/Stan Models/Hybrid_PMF_include_clusters_{include_clusters}_variance_known_{variance_known}')

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Initializations/{chain_id}'
output_dir2 = f'{path}/MAP/{chain_id}'
inits2 = f'{output_dir2}/inits.json'

num_warmup = 1000
num_samples = 100

try:
    # Update initial inits with MAP from previous
    csv_file = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if f.endswith('.csv')][0]
    MAP = cmdstanpy.from_csv(csv_file)
    init = {}
    keys = MAP.stan_variables().keys()
    for key in keys:
        try:
            init[key] = MAP.stan_variables()[key].tolist()
        except:
            init[key] = MAP.stan_variables()[key]
    with open(inits2, 'w') as f:
        json.dump(init, f)
    del csv_file, MAP, init
except:
    pass

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
        prev_csv_files = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if not f.endswith('.json')]
        MAP = model.optimize(data=data_file, output_dir=output_dir2, inits=inits2, show_console=True,  iter=1000000, refresh=1000, 
                        algorithm='lbfgs', jacobian=True, tol_rel_grad=1e-20, tol_rel_obj=1e-20, tol_param=1e-10)
        e=False
        for f in prev_csv_files:
            os.remove(f)
    except:
        delete_files = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if not f.endswith('.json')]
        for f in delete_files:
            os.remove(f)
        e=True
        iter+=1


