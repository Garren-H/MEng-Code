import numpy as np # type: ignore
import json
import os
import cmdstanpy # type: ignore
import sys
import pandas as pd # type: ignore
from multiprocessing import Pool

sys.path.insert(0, '/home/22796002') # include home directory in path to call a python file

# import local function files

from All_code import subsets # python script that extracts data for functional group of interest
import k_means
from generate_stan_model_code import generate_stan_code #type: ignore

# Change this line to match the functional groups to extract
functional_groups = np.array(['Alkane', 'Primary alcohol'])


# get arguments from command line
include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
variance_known = bool(int(sys.argv[2])) # True if you want to use known variance information

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'
path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}'

os.makedirs(path)

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
D = int(N_known/N)
Idx_known = subset_df.iloc[subset_Indices_T[:,0],7:9].to_numpy()

# obtain known variances from NRTL Regression
if variance_known:
    v = []
    for idx in init_indices_T:
        csv_files = [f'/home/22796002/Regression/Results/{idx}/Step3/{f}' for f in os.listdir(f'/home/22796002/Regression/Results/{idx}/Step3/') if f.endswith('.csv')]
        fit = cmdstanpy.from_csv(csv_files)
        max_lp = np.argmax(fit.method_variables()['lp__'].T.flatten())
        v += [fit.v[max_lp]]

# obtain cluster information; first;y obtain number of functional groups to give as maximum to the number of clusters
if include_clusters:
    with pd.ExcelFile("All Data.xlsx") as f:
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
model_code = generate_stan_code(D=D, include_clusters=include_clusters, variance_known=variance_known)

# save stan code to file
with open(f'{path}/Hybrid_PMF.stan', 'w') as f:
    f.write(model_code)

# compile stan code
Hybrid_PMF = cmdstanpy.CmdStanModel(stan_file=f'{path}/Hybrid_PMF.stan', cpp_options={'STAN_THREADS': True})


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

# set number of chains and threads per chain
chains = 8
threads_per_chain = 4

os.environ['STAN_NUM_THREADS'] = str(threads_per_chain)

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Step1'
os.makedirs(output_dir1)
inits1 = f'{output_dir1}/inits.json'
inits = {'v_ARD': np.array([10**(-2*j) for j in range(D)])[::-1].tolist()}
with open(inits1, 'w') as f:
    json.dump(inits, f)

print('Step1: Sampling using random initializations')

fit1 = Hybrid_PMF.sample(data=f'{path}/data.json', output_dir=output_dir1, 
                         chains=chains, parallel_chains=chains,
                         threads_per_chain=threads_per_chain, refresh=1,
                         inits=inits1, iter_warmup=5000, iter_sampling=1000)

# Step 2. Optimizing each chain using initializations from above
output_dir2 = [f'{path}/Step2/{i}' for i in range(chains)]
for d2 in output_dir2:
    os.makedirs(d2)

inits2 = [f'{d2}/inits.json' for d2 in output_dir2]
max_lp = [np.argmax(fit1.method_variables()['lp__'][:,i]) + 1000*i for i in range(chains)]

dict_keys = list(fit1.stan_variables().keys())

for i in range(chains):
    init = {}
    for key in dict_keys:
        try:
            init[key] = fit1.stan_variables()[key][max_lp[i]].tolist()
        except:
            init[key] = fit1.stan_variables()[key][max_lp[i]]
    with open(inits2[i], 'w') as f:
        json.dump(init, f)

print('Step2: Optimizing each chain using initializations from above')
iters = [[output_dir2[i], inits2[i]] for i in range(chains)]

def optimize_chain(output_dir, inits):
    MAP = Hybrid_PMF.optimize(data=f'{path}/data.json', output_dir=output_dir,
                              inits=inits, iter=10000000, algorithm='lbfgs', 
                              refresh=1000, tol_rel_grad=1e-10, tol_param=1e-20, 
                              tol_obj=1e-8)
    return MAP

with Pool(chains) as pool:
    MAP = pool.starmap(optimize_chain, iters)

# Step 3. Sampling using MAP estimates from above

output_dir3 = f'{path}/Step3'
os.makedirs(output_dir3)
inits3 = [f'{output_dir3}/inits_chain{j}.json' for j in range(chains)]
for i in range(chains):
    init = {}
    for key in dict_keys:
        try:
            inits[key] = MAP[i].stan_variables()[key].tolist()
        except:
            inits[key] = MAP[i].stan_variables()[key]
    with open(inits3[i], 'w') as f:
        json.dump(init, f)

print('Step3: Sampling using MAP estimates from above')
fit3 = Hybrid_PMF.sample(data=f'{path}/data.json', output_dir=output_dir3,
                         inits=inits3, refresh=1, iter_warmup=5000, 
                         iter_sampling=1000, chains=chains, parallel_chains=chains, 
                         threads_per_chain=threads_per_chain, max_treedepth=12)
