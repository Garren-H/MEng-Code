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
D = N
Idx_known = subset_df.iloc[subset_Indices_T[:,0],7:9].to_numpy()

try:
    os.makedirs(path)

except:
    print(f'Folder {path} already exists')
    print(f'   Include clusters: {include_clusters}')
    print(f'   Variance known: {variance_known}')
    print('Nothing to be done')
    print('Existing')
    exit()

# obtain known variances from NRTL Regression
if variance_known:
    v_all = json.load(open('/home/ghermanus/lustre/Hybrid PMF/data_model_variance.json'))
    v = np.array(v_all)[init_indices_T].tolist()

# obtain cluster information; first;y obtain number of functional groups to give as maximum to the number of clusters
if include_clusters:
    with pd.ExcelFile("/home/ghermanus/lustre/All Data.xlsx") as f:
        comp_names = pd.read_excel(f, sheet_name='Pure compounds')
        functional_groups = np.sort(functional_groups)
    if functional_groups[0] == 'all':
        # if all return all functional groups
        all_func = comp_names['Functional Group'].to_numpy()
        IUPAC = comp_names['IUPAC'].to_numpy()
    else:
        # else select only neccary functional groups
        idx_name = (comp_names['Functional Group'].to_numpy()[:,np.newaxis] == np.array(functional_groups))
        all_func = np.concatenate([comp_names['Functional Group'][idx_name[:,i]] for i in range(idx_name.shape[1])])
        IUPAC = np.concatenate([comp_names['IUPAC'][idx_name[:,i]] for i in range(idx_name.shape[1])])
    num_funcs = np.unique(all_func).shape[0]
    C_K, Silhouette_K, C_best, K_best = k_means.k_means_clustering(functional_groups, 2, 4*num_funcs)

# generate stan code
model_code = generate_stan_code(include_clusters=include_clusters, variance_known=variance_known)

# save stan code to file
with open(f'{path}/Hybrid_PMF.stan', 'w') as f:
    f.write(model_code)

# compile stan code
model = cmdstanpy.CmdStanModel(stan_file=f'{path}/Hybrid_PMF.stan', cpp_options={'STAN_THREADS': True})

# generate json file:
data = {'N_known': int(N_known),
        'N_points': N_points.tolist(),
        'x': x.tolist(),
        'T': T.tolist(),
        'y': y.tolist(),
        'scaling': scaling.tolist(),
        'a': a,
        'grainsize': int(grainsize),
        'N': int(N),
        'D': int(D),
        'Idx_known': (Idx_known+1).tolist(),}

if variance_known:
    data['v'] = v

if include_clusters:
    data['K'] = int(K_best)
    data['C'] = C_best.tolist()
    data['v_cluster'] = [0.1 for _ in range(K_best)] # may need to change

with open(f'{path}/data.json', 'w') as f:
    json.dump(data, f)

# clear variables defined earlier
del data, model_code, x, T, y, Idx_known, v, N_points, subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T

# set number of chains and threads per chain
chains = 8
threads_per_chain = 3
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Step1'
os.makedirs(output_dir1)

print('Step1: Sampling sort chain using random initialization')
fit = model.sample(data=f'{path}/data.json', output_dir=output_dir1,
                        refresh=100, iter_warmup=10000, 
                        iter_sampling=1000, chains=chains, parallel_chains=chains, 
                        threads_per_chain=threads_per_chain, max_treedepth=5,
                        show_console=True)

# extract max_lp samples per chain as above
output_dir2 = [f'{path}/Step2/{i}' for i in range(chains)]
for d in output_dir2:
    os.makedirs(d)
inits2 = [f'{output_dir2[i]}/inits.json' for i in range(chains)]
max_lp = [np.argmax(fit.method_variables()['lp__'][:,i]) + 1000*i for i in range(chains)]

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
        fit = model.optimize(data=f'{path}/data.json', output_dir=output_dir2[i],
                        inits=inits2[i], show_console=True,  iter=10000000, refresh=1000, 
                        algorithm='lbfgs', jacobian=True, history_size=20, tol_rel_grad=1e-20)
    except:
        fit = None
    return fit

print('Step2: MAP estimation using initializations from sampling')
with Pool(chains) as p:
    fits = p.map(get_MAP, range(chains))

