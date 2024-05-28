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
D = int(sys.argv[3]) # lower rank of feature matrices

print('Evaluating the following conditions for the Hybrid Model:')
print(f'Include clusters: {include_clusters}')
print(f'Variance known: {variance_known}')
print(f'Lower rank of feature matrices: {D}')
print('\n')

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'
path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}'

# get subset of data to work with
subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T = subsets(functional_groups).get_subset_df()

# process dataframe to get relevant json files
x = np.concatenate([subset_df['Composition component 1 [mol/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
T = np.concatenate([subset_df['Temperature [K]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
y = np.concatenate([subset_df['Excess Enthalpy [J/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
N_known = subset_Indices_T.shape[0]
N_points = subset_Indices_T[:,1] + 1 - subset_Indices_T[:,0]
scaling = np.array([1, 1e-3, 1e3, 1])
grainsize = 1
a = 0.3
N = np.max(Info_Indices['Component names']['Index'])+1
D = int(np.min([D, N]))
Idx_known = subset_df.iloc[subset_Indices_T[:,0],7:9].to_numpy()

path += f'/rank_{D}'

# compile stan code
model = cmdstanpy.CmdStanModel(stan_file=f'{path}/Hybrid_PMF.stan', cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 8
threads_per_chain = 3

os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Step1'

print('Step1: Pathfinder using random initializations')

csv_files = [f'{output_dir1}/{f}' for f in os.listdir(output_dir1) if f.endswith('.csv') and not f.endswith('-profile.csv')]

pathfinder = cmdstanpy.from_csv(csv_files)

inits = pathfinder.create_inits(chains=chains)

# Step 2. Optimizing each chain using initializations from above
output_dir2 = f'{path}/Step2'

inits2 = [f'{output_dir2}/inits_{i}.json' for i in range(chains)]

dict_keys = list(inits[0].keys())

for i in range(chains):
    init = {}
    for key in dict_keys:
        try:
            init[key] = inits[i][key].tolist()
        except:
            init[key] = inits[i][key]
    with open(inits2[i], 'w') as f:
        json.dump(init, f)

print('Step2: Sampling using pathfinder inits')
fit = model.sample(data=f'{path}/data.json', output_dir=output_dir2,
                        inits=inits2, refresh=1, iter_warmup=5000, 
                        iter_sampling=1000, chains=chains, parallel_chains=chains, 
                        threads_per_chain=threads_per_chain, max_treedepth=12,
                        metric='dense_e', save_profile=True, sig_figs=18)

