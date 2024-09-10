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

include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
add_zeros = bool(int(sys.argv[2])) # True if we need to add zeros
ARD = bool(int(sys.argv[3])) # True if we need to add ARD
functional_group_string = sys.argv[4] # functional groups
functional_groups = functional_group_string.split(',') # process functional groups into array
chain_id = int(sys.argv[5]) # chain id
num_non_zero_ARD = 1+2*(chain_id+1) # number of non-zero ARD values

# data file
path = f'Subsets/'
path += functional_groups[0]
for functional_group in functional_groups[1:]:
    path += f'_{functional_group}'
data_file = f'{path}/data.json'
path += f'/Include_clusters_{include_clusters}/Add_zeros_{add_zeros}'

#Adjust data
data = json.load(open(data_file, 'r'))
data['D'] = num_non_zero_ARD
if not ARD:
    data['v_ARD'] = np.array([100 for _ in range(num_non_zero_ARD)])

# select stan models with and without ARD
stan_file = f'Stan Models/Pure_PMF_include_clusters_{include_clusters}_zeros_{add_zeros}_ARD_{ARD}.stan'
model = cmdstanpy.CmdStanModel(stan_file=stan_file, cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 1
threads_per_chain = 1

# Total threads for stan to use for parallel computations
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Directories to store output
output_dir2 = f'{path}/MAP/{num_non_zero_ARD}'

# Directory of init file
inits2 = f'{output_dir2}/inits.json'

#update inits file with previous MAP and delete previous MAP files
try:
    csv_file = [f'{output_dir2}/{file}' for file in os.listdir(output_dir2) if file.endswith('.csv')][0]
    MAP = cmdstanpy.from_csv(csv_file)
    init = {}
    keys = list(MAP.stan_variables().keys())
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

#delete previous MAP files
try:
    del_files = [f'{output_dir2}/{file}' for file in os.listdir(output_dir2) if not file.endswith('.json')]
    for file in del_files:
        os.remove(file)
except:
    pass


print('Running Pure PMF Model with the following conditions:')
print(f'Include clusters: {include_clusters}')
print(f'Zeros: {add_zeros}')
print(f'ARD: {False}')
# Run MAP step
try:
    print(f'Initial ARD values: {json.load(open(inits2, 'r'))['v_ARD']}')
except:
    print(f'Initial ARD values: {data['v_ARD'].tolist()}')

MAP = model.optimize(data=data, inits=inits2, show_console=True,  iter=1000000, refresh=1000, 
                algorithm='lbfgs', jacobian=False, tol_rel_grad=1e-20, tol_rel_obj=1e-20, tol_param=1e-10,
                tol_grad=1e-20, output_dir=output_dir2, init_alpha=1e-20)

try:
    print(f'Final MAP values{MAP.v_ARD.tolist()}')
except:
    print(f'Final MAP values{data['v_ARD'].tolist()}')