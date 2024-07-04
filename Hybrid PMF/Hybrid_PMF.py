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

# Change this line to match the functional groups to extract
functional_groups = np.array(['Alkane', 'Primary alcohol'])

# get arguments from command line
include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
variance_known = bool(int(sys.argv[2])) # True if you want to use known variance information

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

os.makedirs(path)

# set number of chains and threads per chain
chains = 8
threads_per_chain = 3
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Obtain stan model
model = cmdstanpy.CmdStanModel(exe_file=f'/home/ghermanus/lustre/Hybrid PMF/Stan Models/Hybrid_PMF_include_clusters_{include_clusters}_variance_known_{variance_known}')

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Initializations'
steps = 5
num_warmup = 2000
num_samples = 100
output_dir1 = [f'{output_dir1}/{i}' for i in range(steps)]

print('Step1: Sampling sort chain using random initialization')
for i in range(steps):
    os.makedirs(output_dir1[i])
    print(f'Iter {i+1} out of {steps}')
    if i == 0:
        fit = model.sample(data=data_file, output_dir=output_dir1[i],
                                refresh=100, iter_warmup=num_warmup, 
                                iter_sampling=num_samples, chains=chains, parallel_chains=chains, 
                                threads_per_chain=threads_per_chain, max_treedepth=5)
    else:
        #save inits from previous step
        max_lp = [np.argmax(fit.method_variables()['lp__'][:,i]) + num_samples*i for i in range(chains)]
        dict_keys = list(fit.stan_variables().keys())
        inits1 = []
        for j in range(chains):
            init = {}
            for key in dict_keys:
                try:
                    init[key] = fit.stan_variables()[key][max_lp[j]].tolist()
                except:
                    init[key] = fit.stan_variables()[key][max_lp[j]]
            inits1 += [init]
        # perform sampling again, with stepsize and metric reset
        fit = model.sample(data=data_file, output_dir=output_dir1[i],
                                refresh=100, iter_warmup=num_warmup, 
                                iter_sampling=num_samples, chains=chains, parallel_chains=chains, 
                                threads_per_chain=threads_per_chain, max_treedepth=5, inits=inits1)
        del inits1, init

# extract max_lp samples per chain as above
output_dir2 = [f'{path}/MAP/{i}' for i in range(chains)]
for d in output_dir2:
    os.makedirs(d)
inits2 = [f'{output_dir2[i]}/inits.json' for i in range(chains)]
max_lp = [np.argmax(fit.method_variables()['lp__'][:,i]) + num_samples*i for i in range(chains)]

dict_keys = list(fit.stan_variables().keys())

for i in range(chains):
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

# Step 2. Run MAP estimation using initializations from sampling
def get_MAP(i):
    try:
        fit = model.optimize(data=data_file, output_dir=output_dir2[i],
                        inits=inits2[i], show_console=True,  iter=10000000, refresh=1000, 
                        algorithm='lbfgs', jacobian=True, history_size=20, tol_rel_grad=1e-20, tol_rel_obj=1e-20)
    except:
        fit = None
    return fit

print('Step2: MAP estimation using initializations from sampling')
with Pool(chains) as p:
    fits = p.map(get_MAP, range(chains))

