import numpy as np # type: ignore
import json
import os
import pandas as pd # type: ignore
import sys

sys.path.insert(0, '/home/ghermanus/lustre') # include home directory in path to call a python file

# import local function files

from All_code import subsets # python script that extracts data for functional group of interest
import k_means

func_groups_string = sys.argv[1] # functional groups to extract
functional_groups = func_groups_string.split('.') # restructure to numpy array 

# create file to store stan models and results
path = 'Subsets/'

for functional_group in functional_groups:
    if path == 'Subsets/':
        path += f'{functional_group}'
    else:
        path += f'_{functional_group}'

try:
    os.makedirs(path)

except:
    print(f'Folder {path} already exists')
    print('Nothing to be done')
    print('Existing')
    exit()

# get subset of data to work with
subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T = subsets(functional_groups).get_subset_df()

# process dataframe to get relevant json files
x = np.concatenate([subset_df['Composition component 1 [mol/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
T = np.concatenate([subset_df['Temperature [K]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
y = np.concatenate([subset_df['Excess Enthalpy [J/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
N_known = subset_Indices_T.shape[0]
N_points = subset_Indices_T[:,1] + 1 - subset_Indices_T[:,0]
scaling = np.array([1, 1e-3, 1e2, 1])
grainsize = 1
a = 0.3
N = np.max(Info_Indices['Component names']['Index'])+1
D = np.min([N,20])
Idx_known = subset_df.iloc[subset_Indices_T[:,0],7:9].to_numpy()

# obtain known data variance
v_all = json.load(open('/home/ghermanus/lustre/Hybrid PMF/data_model_variance.json'))
v = np.array(v_all)[init_indices_T].tolist()

# obtain cluster information; first;y obtain number of functional groups to give as maximum to the number of clusters
with pd.ExcelFile("/home/ghermanus/lustre/All Data.xlsx") as f:
    comp_names = pd.read_excel(f, sheet_name='Pure compounds')
    functional_groups = np.sort(functional_groups)
    CA = comp_names['Self Cluster assignment'].to_numpy()
    fg = comp_names['Functional Group'].to_numpy()
    
    if functional_groups[0] == 'all':
        CA = CA
    else:
        idx = np.zeros(len(fg)).astype(int)
        for i in functional_groups:
            idx += (fg == i).astype(int)
        idx = idx.astype(bool)
        CA = CA[idx]
        del idx

    unique_CA = np.unique(CA)
    C = (unique_CA[:,np.newaxis] == CA[np.newaxis,:]).astype(int)
    K = len(unique_CA)

    del comp_names, CA, fg

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
        'Idx_known': (Idx_known+1).tolist(),
        'scale_cauchy': 1e-30}

data['v'] = v

data['K'] = int(K)
data['C'] = C.tolist()
data['v_cluster'] = [0.1**2 for _ in range(K)] # may need to change

data['T_zeros'] = np.array([288.15, 298.15, 308.15]).tolist()
data['x_zeros'] = np.array([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
data['N_x_zeros'] = len(data['x_zeros'])
data['N_T_zeros'] = len(data['T_zeros'])
data['sigma_zeros'] = 15

with open(f'{path}/data.json', 'w') as f:
    json.dump(data, f)
