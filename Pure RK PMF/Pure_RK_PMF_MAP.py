import cmdstanpy # type: ignore
import json
import sys
import os
import random
import numpy as np #type: ignore
import pandas as pd #type: ignore
import shutil
from multiprocessing import Pool

sys.path.append('/home/22796002')

import All_code

# set random seed
random.seed(1)

ncpu=8

# Set functional groups to evaulate
functional_groups = np.array(['Alkane', 'Primary alcohol'])

# Obtain subset data for processing
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
D = 10
order = 3

# Assign data
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
               'v_D': (1e-3*np.ones((N_known+N_unknown))).tolist()}

# Create folder to store:
folder = 'Subsets/Constant_var/MAP'

for func in functional_groups:
    if folder == 'Subsets':
        folder = f'{folder}/{func}'
    else:
        folder = f'{folder}_{func}'

os.makedirs(folder)

# Save data to json file
with open(f'{folder}/data.json', 'w') as f:
    json.dump(Hybrid_Data, f)

# define function to for MAP
def do_MAP(output_dir):
    seed = random.randint(1, 2**32+1)
    try:
        MAP = cmdstanpy.CmdStanModel(exe_file='Stan Models/PMF_RK_GP_const_var').optimize(data=f'{folder}/data.json',
                                                                        iter=1000000, seed=seed,
                                                                        refresh=1, jacobian=True,
                                                                        output_dir=output_dir,
                                                                        show_console=True,
                                                                        algorithm='lbfgs', tol_rel_grad=1,
                                                                        )
    except:
        shutil.rmtree(output_dir)
        try:
            MAP = cmdstanpy.CmdStanModel(exe_file='Stan Models/PMF_RK_GP_const_var').optimize(data=f'{folder}/data.json',
                                                                            iter=1000000, seed=seed,
                                                                            refresh=1, jacobian=True,
                                                                            output_dir=output_dir,
                                                                            show_console=True,
                                                                            algorithm='newton'
                                                                            )
        except:
            shutil.rmtree(output_dir)
            MAP = []
            print(f'{output_dir}:')
            print('Could not optimize function. Try specifying intital values')
    return MAP

output_dirs = [f'{folder}/{j}' for j in range(ncpu)]
with Pool(ncpu) as pool:
    MAP = pool.map(do_MAP, output_dirs)

