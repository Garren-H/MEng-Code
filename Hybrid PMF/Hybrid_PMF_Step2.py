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


# compile stan code
model = cmdstanpy.CmdStanModel(exe_file=f'/home/ghermanus/lustre/Hybrid PMF/Stan Models/Hybrid_PMF_include_clusters_{include_clusters}_variance_known_{variance_known}')

# set number of chains and threads per chain
chains = 8
threads_per_chain = 3

os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Sampling'
inits1 = f'{output_dir1}/inits.json'

try:
    os.makedirs(output_dir1)
except:
    print(f'Directory {output_dir1} already exists')

# Obtain inits corresponding to max_lp

csv_files = [f'{path}/MAP/{i}/{f}' for i in np.sort(os.listdir(f'{path}/MAP')) for f in os.listdir(f'{path}/MAP/{i}') if f.endswith('.csv')]

MAP = []
for csv_file in csv_files:
    try:
        MAP += [cmdstanpy.from_csv(csv_file)]
    except:
        print(f'Faulty file: {csv_file}')

max_lp = np.argmax([map.optimized_params_dict['lp__'] for map in MAP])
keys = list(MAP[0].stan_variables().keys())

init = {}
for key in keys:
    try:
        init[key] = MAP[max_lp].stan_variables()[key].tolist()
    except:
        init[key] = MAP[max_lp].stan_variables()[key]
with open(inits1, 'w') as f:
    json.dump(init, f)

# clear memory
del MAP, csv_files, keys, init, max_lp

print('Sampling using MAP inits')
fit = model.sample(data=data_file, output_dir=output_dir1,
                        inits=inits1, refresh=10, iter_warmup=5000, 
                        iter_sampling=1000, chains=chains, parallel_chains=chains, 
                        threads_per_chain=threads_per_chain, max_treedepth=10,
                        metric='diag_e', sig_figs=18)

