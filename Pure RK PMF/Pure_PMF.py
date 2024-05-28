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
D = int(sys.argv[4]) # lower rank of feature matrices

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

# Extract data from dataframe
x = np.concatenate([subset_df['Composition component 1 [mol/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
T = np.concatenate([subset_df['Temperature [K]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
y = np.concatenate([subset_df['Excess Enthalpy [J/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
N_known = subset_Indices_T.shape[0]
N_points = subset_Indices_T[:,1] + 1 - subset_Indices_T[:,0]
grainsize = 1
N = np.max(Info_Indices['Component names']['Index'])+1
D = np.min([D, N])
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

path += f'/rank_{D}'

try:
    os.makedirs(path)

except:
    print(f'Folder {path} already exists')
    print(f'Lower rank of {D} already evaluated for conditions:')
    print(f'   Include clusters: {include_clusters}')
    print(f'   Variance known: {variance_known}')
    print(f'   Variance MC known: {variance_MC_known}')
    print('Nothing to be done')
    print('Existing')
    exit()


# save data to json file
data_file = f'{path}/data.json'
with open(data_file, 'w') as f:
    json.dump(data, f)

# Get stan code
stan_code = generate_stan_code(include_clusters=include_clusters, 
                            variance_known=variance_known, 
                            variance_MC_known=variance_MC_known)

with open(f'{path}/Pure_PMF.stan', 'w') as f:
    f.write(stan_code)

# compile stan model
model = cmdstanpy.CmdStanModel(stan_file=f'{path}/Pure_PMF.stan', cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 4
threads_per_chain = 4

os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Step 1. Run sampling with random initialzations, but ARD variances initialized
output_dir1 = f'{path}/Step1'
os.makedirs(output_dir1)

print('Step1: Sampling sort chain using random initialization')
try:
    fit = model.sample(data=f'{path}/data.json', output_dir=output_dir1,
                            refresh=100, iter_warmup=100, 
                            iter_sampling=100, chains=chains, parallel_chains=chains, 
                            threads_per_chain=threads_per_chain, max_treedepth=5,
                            metric='dense_e', save_profile=True, sig_figs=18)
except: # If some chains failed:
    print('Error in sampling')
    print('Some chains may have failed')
    print('Validating the chains individually:')
    csv_files = [f'{output_dir1}/{f}' for f in os.listdir(output_dir1) if f.endswith('.csv')]
    failed_idx = []
    for i in range(chains):
        try:
            fit = cmdstanpy.from_csv(csv_files[i])
        except:
            failed_idx += [i] 
    print(f'Failed chains: {failed_idx}')
    print('Removing failed chains')
    for idx in failed_idx:
        os.remove(csv_files[idx])
    csv_files = [f'{output_dir1}/{f}' for f in os.listdir(output_dir1) if f.endswith('.csv') and not f.endswith('profile.csv')]
    fit = from_csv(csv_files)

    if sum(failed_idx) == 0:
        print('No chains failed')
        print('Something may have gone wrong with cmdstanpy post-processing')
        print('Continuing')

# extract max_lp samples per chain as above
output_dir2 = [f'{path}/Step2/{i}' for i in range(chains)]
inits2 = [f'{d}/inits.json' for d in output_dir2]

# Correctly get max_lp when some chains failed
try: # Assign max_lp for each chain if no chains failed
    max_lp = [np.argmax(fit.method_variables()['lp__'][:,i]) + 1000*i for i in range(chains)]
except: # Assign failed chains to overall max_lp
    max_lp = [np.argmax(fit.stan_variables()['lp__'][:,i]) + 1000*i for i in range(len(csv_files))]
    overall_max_lp = np.argmax(fit.stan_variables()['lp__'].T.flatten())
    print('Assinging failed chains to overall max_lp')
    for i in range(chains-len(csv_files)): 
        max_lp += [overall_max_lp]

dict_keys = list(fit.stan_variables().keys())

# save inits
for i in range(chains):
    os.makedirs(output_dir2[i])
    init = {}
    for key in dict_keys:
        try:
            init[key] = fit.stan_variables()[key][max_lp[i]].tolist()
        except:
            init[key] = fit.stan_variables()[key][max_lp[i]]
    with open(inits2[i], 'w') as f:
        json.dump(init, f)

# Step 2. Run optimization using initializations given above
def run_optimization(output_dir, inits):
    try:
        fit = model.optimize(data=f'{path}/data.json', inits=inits, output_dir=output_dir, 
                        save_profile=True, sig_figs=18, algorithm='lbfgs', 
                        tol_rel_grad=1e-20, show_console=True, jacobian=True,
                        refresh=100, iter=10000000)
    except:
        fit = ['Failed chain']
    return fit

iters = [[output_dir2[i], inits2[i]] for i in range(chains)]

with Pool(chains) as pool:
    MAP = pool.starmap(run_optimization, iters)