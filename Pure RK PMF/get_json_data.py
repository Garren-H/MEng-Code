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

# create directory if it does not exist
if not os.path.exists(path):
    os.makedirs(path)

# Extract data from dataframe
x = np.concatenate([subset_df['Composition component 1 [mol/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
T = np.concatenate([subset_df['Temperature [K]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
y = np.concatenate([subset_df['Excess Enthalpy [J/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
N_known = subset_Indices_T.shape[0]
N_points = subset_Indices_T[:,1] + 1 - subset_Indices_T[:,0]
grainsize = 1
N = np.max(Info_Indices['Component names']['Index'])+1
D = np.min([20, N])
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
        'v': v.tolist(),
        'scale_lower': 1e-2}

data['K'] = int(K_best)
data['C'] = C_best.tolist()
data['v_cluster'] = [0.1 for _ in range(K_best)] # may need to change

# save data to json file
data_file = f'{path}/data.json'
with open(data_file, 'w') as f:
    json.dump(data, f)

