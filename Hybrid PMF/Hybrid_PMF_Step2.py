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
variance_known = bool(int(sys.argv[2])) # True if you want to use known variance information
func_groups_string = sys.argv[3] # functional groups to extract
functional_groups = func_groups_string.split(',') # restructure to numpy array 

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
model = cmdstanpy.CmdStanModel(exe_file=f'/home/ghermanus/lustre/Hybrid PMF/Stan Models/Hybrid_PMF_include_clusters_{include_clusters}_variance_known_{variance_known}', cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 5
threads_per_chain = 4

os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Sampling'
inits1 = f'{output_dir1}/inits.json'

try:
    os.makedirs(output_dir1)
except:
    print(f'Directory {output_dir1} already exists')

# Obtain inits corresponding to max_lp
csv_files = [f'{path}/MAP/{i}' for i in np.sort(os.listdir(f'{path}/MAP')) if i.isdigit()]

MAP = []
for i in range(len(csv_files)):
    try:
        csv_file = f'{csv_files[i]}/{[f for f in os.listdir(csv_files[i]) if f.endswith('.csv')][0]}'
        MAP += [cmdstanpy.from_csv(csv_file)]
    except:
        try: 
            MAP += [f'{csv_files[i]}/inits.json']
        except:
            print(f'Faulty csv and json file in {csv_files[i]}')
    try:
        del csv_file
    except:
        print('')

lp = []
for map in MAP:
    try:
        lp += [map.optimized_params_dict['lp__']]
    except:
        lp += [model.log_prob(data=data_file, params=map).iloc[0,0]]

max_lp = np.argmax(lp)

try:
    keys = list(MAP[0].stan_variables().keys())
except:
    keys = list(json.load(open(MAP[0], 'r')).keys())

init = {}
for key in keys:
    try:
        init[key] = MAP[max_lp].stan_variables()[key].tolist()
    except:
        try:
            init[key] = MAP[max_lp].stan_variables()[key]
        except:
            try:
                init[key] = json.load(open(MAP[max_lp],'r'))[key].tolist()
            except:
                init[key] = json.load(open(MAP[max_lp],'r'))[key]
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

