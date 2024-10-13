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

# get arguments from command line
include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
include_zeros = bool(int(sys.argv[2])) # True if you want to use known variance information
ARD = bool(int(sys.argv[3])) # True if you want to use ARD
func_groups_string = sys.argv[4] # functional groups to extract
functional_groups = func_groups_string.split('.') # restructure to numpy array 
chain_id = int(sys.argv[5]) # chain id

num_non_zero_ARD = 2+chain_id*3 # number of non-zero ARD values

print('Evaluating the following conditions for the Hybrid Model:')
print(f'Include clusters: {include_clusters}')
print(f'Include zeros: {include_zeros}')
print(f'ARD: {ARD}')
print('\n')

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'
data_file = f'{path}/data.json' # file where data is stored
path += f'/Include_clusters_{include_clusters}/include_zeros_{include_zeros}/ARD_{ARD}'

# Adjust data to reflect number of non-zero ARD variances
data = json.load(open(data_file, 'r'))
data['D'] = int(num_non_zero_ARD)
if not ARD:
    data['v_ARD'] = np.array([0.1 for _ in range(num_non_zero_ARD)])

# set number of chains and threads per chain
chains = 6
threads_per_chain = 4
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Obtain stan model
model = cmdstanpy.CmdStanModel(exe_file=f'/home/ghermanus/lustre/Hybrid PMF Adj/Stan Models/Hybrid_PMF_include_clusters_{include_clusters}_include_zeros_{include_zeros}_ARD_{ARD}')

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir2 = f'{path}/Sampling/{num_non_zero_ARD}'

# inits directory
prev_csv_files = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if f.endswith('.csv')]
prev_txt_files = [f'{output_dir2}/{f}' for f in os.listdir(output_dir2) if f.endswith('.txt')]
fit_prev = cmdstanpy.from_csv(prev_csv_files)
inits = [{} for _ in range(chains)]
metric = [{} for _ in range(chains)]
step_size = fit_prev.step_size.tolist()
keys = list(fit_prev.stan_variables().keys())
for i in range(chains):
    for key in keys:
        inits[i][key] = np.mean(fit_prev.stan_variables()[key][i*chains:(i+1)*chains], axis=0)

    metric[i]['inv_metric'] = fit_prev.metric[i]
del fit_prev, keys

try:
    print(f'Initial ARD values: {inits[0]['v_ARD']}')
except:
    print(f'Initial ARD values: {data['v_ARD'].tolist()}')


fit = model.sample(data=data, inits=inits, output_dir=output_dir2, 
                   iter_sampling=2000, iter_warmup=1000, max_treedepth=11,
                   chains=chains, threads_per_chain=threads_per_chain, refresh=10,
                   force_one_process_per_chain=True, metric=metric, step_size=step_size)

for f in prev_txt_files:
    os.remove(f)
for f in prev_csv_files:
    os.remove(f)

