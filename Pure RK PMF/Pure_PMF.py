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

# Change this line to match the functional groups to extract
functional_groups = np.array(['Alkane', 'Primary alcohol'])

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'

data_file = f'{path}/data.json'

path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}/Variance_MC_known_{variance_MC_known}'

try:
    os.makedirs(path)

except:
    print(f'Folder {path} already exists')
    print(f'Include clusters: {include_clusters}')
    print(f'Variance known: {variance_known}')
    print(f'Variance MC known: {variance_MC_known}')
    print('Nothing to be done')
    print('Exiting...')
    exit()

# compile stan model
exe_file = f'/home/22796002/Pure RK PMF/Stan Models/Pure_PMF_Include_clusters_{include_clusters}_Variance_known_{variance_known}_Variance_MC_known_{variance_MC_known}'
model = cmdstanpy.CmdStanModel(exe_file=exe_file, cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 4
threads_per_chain = 4

os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Step1'
os.makedirs(output_dir1)

print('Step1: Sampling sort chain using random initialization')
num_warmup = 10000
num_samples = 100
fit = model.sample(data=data_file, output_dir=output_dir1,
                        refresh=100, iter_warmup=num_warmup, 
                        iter_sampling=num_samples, chains=chains, parallel_chains=chains, 
                        threads_per_chain=threads_per_chain, max_treedepth=5,
                        sig_figs=18)

# extract max_lp samples per chain as above
output_dir2 = [f'{path}/Step2/{i}' for i in range(chains)]
inits2 = [f'{d}/inits.json' for d in output_dir2]

# Correctly get max_lp when some chains failed
max_lp = [np.argmax(fit.method_variables()['lp__'][:,i]) + num_samples*i for i in range(chains)]

dict_keys = list(fit.stan_variables().keys())

# save inits
for i in range(chains):
    os.makedirs(output_dir2[i])
    init = {}
    for key in dict_keys:
        try:
            init[key] = fit.stan_variables()[key][max_lp[i]].tolist()
        except:
            init[key] = fit.stan_variables()[key][max_lp[i]]
    with open(inits2[i], 'w') as f:
        json.dump(init, f)

# clear variables from memory
del fit, init

# Step 2. Run optimization using initializations given above
def run_optimization(output_dir, inits):
    try:
        fit = model.optimize(data=data_file, inits=inits, output_dir=output_dir, 
                        save_profile=True, sig_figs=18, algorithm='lbfgs', 
                        tol_rel_grad=1e-20, show_console=True, jacobian=True,
                        refresh=1000, iter=10000000)
    except:
        fit = ['Failed chain']
    return fit

iters = [[output_dir2[i], inits2[i]] for i in range(chains)]

with Pool(chains) as pool:
    MAP = pool.starmap(run_optimization, iters)