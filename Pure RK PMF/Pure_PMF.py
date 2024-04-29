import numpy as np # type: ignore
import json
import cmdstanpy # type: ignore
import sys
import os
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

# get subset of data to work with
subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T = subsets(functional_groups).get_subset_df()

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'
path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}/Variance_MC_known_{variance_MC_known}'

os.makedirs(path)

# Extract data from dataframe
x = np.concatenate([subset_df['Composition component 1 [mol/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
T = np.concatenate([subset_df['Temperature [K]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
y = np.concatenate([subset_df['Excess Enthalpy [J/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
N_known = subset_Indices_T.shape[0]
N_points = subset_Indices_T[:,1] + 1 - subset_Indices_T[:,0]
grainsize = 1
N = np.max(Info_Indices['Component names']['Index'])+1
D = int(N_known/N)
Idx_known = subset_df.iloc[subset_Indices_T[:,0],7:9].to_numpy()
Idx_unknown = np.array([[i, j] for i in range(N) for j in range(i+1,N)])
idx = np.sum(np.char.add(Idx_unknown[:,0].astype(str), Idx_unknown[:,1].astype(str))[:,np.newaxis] ==
             np.char.add(Idx_known[:,0].astype(str), Idx_known[:,1].astype(str))[np.newaxis,:], axis=1) == 0
Idx_unknown = Idx_unknown[idx,:]
N_unknown = int((N**2-N)/2 - N_known)
v = 1e-3*np.ones(N_known)
v_MC = 1
T2_int = np.array([293.15, 298.15])
x2_int = np.linspace(0, 1, 21)[1:-1]
N_T = T2_int.shape[0]
N_C = x2_int.shape[0]
order = 3
jitter = 1e-7

# obtain cluster information; first;y obtain number of functional groups to give as maximum to the number of clusters
if include_clusters:
    with pd.ExcelFile("/home/22796002/All Data.xlsx") as f:
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


# save data in dictionary
data = {'N_known': int(N_known),
        'N_unknown': int(N_unknown),
        'N_points': N_points.tolist(),
        'order': int(order),
        'x1': x.tolist(),
        'T1': T.tolist(),
        'y1': y.tolist(),
        'N_T': int(N_T),
        'N_C': int(N_C),
        'T2_int': T2_int.tolist(),
        'x2_int': x2_int.tolist(),
        'v_MC': v_MC,
        'grainsize': int(grainsize),
        'N': int(N),
        'D': int(D),
        'Idx_known': (Idx_known+1).tolist(),
        'Idx_unknown': (Idx_unknown+1).tolist(),
        'jitter': jitter,
        'v': v.tolist()
        }

if include_clusters:
    data['K'] = int(K_best)
    data['C'] = C_best.tolist()
    data['v_cluster'] = [0.1 for _ in range(K_best)] # may need to change

# save data to json file
data_file = f'{path}/data.json'
with open(data_file, 'w') as f:
    json.dump(data, f)

# Get stan code
stan_code = generate_stan_code(D=D, include_clusters=include_clusters, 
                               variance_known=variance_known, 
                               variance_MC_known=variance_MC_known)

with open(f'{path}/Pure_PMF.stan', 'w') as f:
    f.write(stan_code)

# compile stan model
model = cmdstanpy.CmdStanModel(stan_file=f'{path}/Pure_PMF.stan', cpp_options={'STAN_THREADS': True})

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

fit1 = model.sample(data=f'{path}/data.json', output_dir=output_dir1, 
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
    MAP = model.optimize(data=f'{path}/data.json', output_dir=output_dir,
                              inits=inits, iter=10000000, algorithm='lbfgs', 
                              refresh=100, tol_rel_grad=1e-10, tol_param=1e-20, 
                              tol_obj=1e-8, show_console=True)
    return MAP

with Pool(chains) as pool:
    MAP = pool.starmap(optimize_chain, iters)