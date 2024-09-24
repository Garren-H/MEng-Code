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

#Adjust data
data = json.load(open(data_file, 'r'))
data['D'] = num_non_zero_ARD
if not ARD:
    data['v_ARD'] = np.array([0.1 for _ in range(num_non_zero_ARD)])

# set number of chains and threads per chain
chains = 1
threads_per_chain = 4
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Obtain stan models
model = cmdstanpy.CmdStanModel(exe_file=f'/home/ghermanus/lustre/Hybrid PMF Adj/Stan Models/Hybrid_PMF_include_clusters_{include_clusters}_include_zeros_{include_zeros}_ARD_{ARD}')

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir2 = f'{path}/MAP/{num_non_zero_ARD}'

# inits directory
inits2 = f'{output_dir2}/inits.json'

# update inits file based on previous
try:
    csv_file = [f'{output_dir2}/{file}' for file in os.listdir(output_dir2) if file.endswith('.csv')][0]
    MAP = cmdstanpy.from_csv(csv_file)
    keys = list(MAP.stan_variables().keys())
    init = {}
    for key in keys:
        try:
            init[key] = MAP.stan_variables()[key].tolist()
        except:
            init[key] = MAP.stan_variables()[key]
    with open(inits2, 'w') as f:
        json.dump(init, f)
    del MAP, init
except:
    try:
        csv_files = [f'{output_dir2}/{i}/{f}' for i in os.listdir(output_dir2) if i.isdigit() for f in os.listdir(f'{output_dir2}/{i}') if f.endswith('.csv')]
        lp = []
        MAP = []
        for file in csv_files:
            try:
                MAP += [cmdstanpy.from_csv(file)]
                lp += [MAP[-1].optimized_params_dict['lp__']]
            except:
                pass
        max_lp = np.argmax(lp)
        MAP = MAP[max_lp]
        keys = list(MAP.stan_variables().keys())
        init = {}
        for key in keys:
            try:
                init[key] = MAP.stan_variables()[key].tolist()
            except:
                init[key] = MAP.stan_variables()[key]
        with open(inits2, 'w') as f:
            json.dump(init, f)
        del MAP, init
    except:
        pass

# delete previous MAP files
try:
    del_files = [f'{output_dir2}/{file}' for file in os.listdir(output_dir2) if file.endswith('.csv') or file.endswith('.txt')]
    for file in del_files:
        os.remove(file)
except:
    pass

# Run MAP step
try:
    print(f'Initial ARD values: {json.load(open(inits2, 'r'))['v_ARD']}')
except:
    print(f'Initial ARD values: {data['v_ARD'].tolist()}')

MAP = model.optimize(data=data, inits=inits2, show_console=True,  iter=1000000, refresh=1000, 
                algorithm='lbfgs', jacobian=False, tol_rel_grad=1e-20, tol_rel_obj=1e-20, tol_param=1e-10,
                tol_grad=1e-20, output_dir=output_dir2, init_alpha=1e-15)

try:
    print(f'Final MAP values{MAP.v_ARD.tolist()}')
except:
    print(f'Final MAP values{data['v_ARD'].tolist()}')