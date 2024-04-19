import sys

sys.path.append('/home/22796002')

import All_code
import pandas as pd #type: ignore
import numpy as np #type: ignore
import json
import os
import cmdstanpy #type: ignore

# change functional group
functional_groups = np.array(['Alkane', 'Primary alcohol'])

folder = 'Subsets'

for func in functional_groups:
    if folder == 'Subsets':
        folder = f'{folder}/{func}'
    else:
        folder = f'{folder}_{func}'

os.makedirs(folder)

subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T = All_code.subsets(functional_groups).get_subset_df()

N = np.max(Info_Indices['Component names']['Index']) + 1
y = []
x = []
T = []
N_points = []
N_known = subset_Indices_T.shape[0]
N_unknown = int(N*(N-1)/2) - N_known
Idx_known = []
Idx_unknown = [[l, k] for l in range(1, N+1) for k in range(l+1,N+1)]

for i in range(N_known):
    x.extend(subset_df['Composition component 1 [mol/mol]'][subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].to_list())
    T.extend(subset_df['Temperature [K]'][subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].to_list())
    y.extend(subset_df['Excess Enthalpy [J/mol]'][subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].to_list())
    N_points += [subset_Indices_T[i,1]+1 - subset_Indices_T[i,0]]
    Idx_known += [[subset_df['Component 1 - Index'][subset_Indices_T[i,0]] + 1, subset_df['Component 2 - Index'][subset_Indices_T[i,0]]+1]]

unknown_string = np.char.add(np.array(Idx_unknown).astype(str)[:,0], np.array(Idx_unknown).astype(str)[:,1])
known_string = np.char.add(np.array(Idx_known).astype(str)[:,0], np.array(Idx_known).astype(str)[:,1])
remove_string = np.sum(unknown_string[:, np.newaxis] == known_string[np.newaxis,:], axis=1).astype(bool)
Idx_unknown = np.delete(np.array(Idx_unknown), remove_string, axis=0).tolist()

x2_int = np.linspace(0, 1, 21)
x2_int = np.delete(np.delete(x2_int, 0), -1).tolist()
N_C = int(len(x2_int))
T2_int = [288.15, 298.15, 308.15]
N_T = int(len(T2_int))

D = 5
order = 3
alpha_lower = 0.40
alpha_upper = 0.60

Hybrid_Data = {'N_known': int(N_known),
               'N_unknown': int(N_unknown),
               'N_points': np.array(N_points).astype(int).tolist(),
               'x1': x,
               'T1': T,
               'y1': y,
               'N_C': int(N_C),
               'N_T': int(N_T),
               'x2_int': x2_int,
               'T2_int': T2_int,
               'Idx_known': np.array(Idx_known).astype(int).tolist(),
               'Idx_unknown': np.array(Idx_unknown).astype(int).tolist(),
               'N': int(N),
               'D': int(D),
               'order': int(order),
               'alpha_lower': alpha_lower,
               'alpha_upper': alpha_upper}

with open(f'{folder}/data.json', 'w') as f:
    json.dump(Hybrid_Data, f)

print("Starting Sampling")

HYBRID_FIT = cmdstanpy.CmdStanModel(stan_file='Stan Models/PMF_HYBRID_WITH_RK.stan').sample(data=f'{folder}/data.json', 
                          chains=8, parallel_chains=8, 
                          iter_warmup=10000, iter_sampling=10000,
                          max_treedepth=10, adapt_delta=0.99,
                          output_dir=folder)

print("Completed Sampling")