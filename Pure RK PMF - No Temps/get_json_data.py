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

from All_code import subsets
import k_means

def convert_to_int(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_to_int(value)
        elif isinstance(value, (np.integer, int)):
            d[key] = int(value)
        elif isinstance(value, list):
            d[key] = [int(item) if isinstance(item, (np.integer, int)) else item for item in value]

func_groups_string = sys.argv[1] # functional groups to extract
functional_groups = func_groups_string.split(',') # restructure to numpy array 

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
v_MC = 0.2
x2_int = np.concatenate([np.append(np.linspace(0,0.45, 10)[1:], [0.495, 1-0.495]), np.linspace(0.55, 1, 10)[:-1]])
N_C = x2_int.shape[0]
order = 3
jitter = 1e-7

# obtain cluster information; firstly obtain number of functional groups to give as maximum to the number of clusters
with pd.ExcelFile("/home/22796002/All Data.xlsx") as f:
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

# save data in dictionary
data = {'N_known': int(N_known),
        'N_unknown': int(N_unknown),
        'N_points': N_points.tolist(),
        'order': int(order),
        'x1': x.tolist(),
        'T1': T.tolist(),
        'y1': y.tolist(),
        'N_C': int(N_C),
        'T2_int': 298.15,
        'x2_int': x2_int.tolist(),
        'v_MC': v_MC,
        'grainsize': int(grainsize),
        'N': int(N),
        'D': int(D),
        'Idx_known': (Idx_known+1).tolist(),
        'Idx_unknown': (Idx_unknown+1).tolist(),
        'jitter': jitter,
        'v': v.tolist(),
        'scale_cauchy': 1e-30}

data['K'] = int(K)
data['C'] = C.tolist()
data['v_cluster'] = [(0.1)**2 for _ in range(K)] # may need to change

data['sigma_refT'] = [0.1, 1e-5, 0.1]

# Update data to only extract temperatures at 298.15
x1 = []
T1 = []
y1 = []
N_points = []
v = []
Idx_known = []

N_known = data['N_known']

for i in range(N_known):
    idx_start = np.sum(data['N_points'][:i]).astype(int)
    idx_end = np.sum(data['N_points'][:i+1]).astype(int)
    xx = data['x1'][idx_start:idx_end]
    tt = data['T1'][idx_start:idx_end]
    yy = data['y1'][idx_start:idx_end]

    t_idx = np.abs(np.array(tt)-298.15)<=0.5

    if np.sum(t_idx) > 0:
        x1 += np.array(xx)[t_idx].tolist()
        T1 += (298.15*np.ones(np.sum(t_idx).astype(int))).tolist()
        y1 += np.array(yy)[t_idx].tolist()
        v += [data['v'][i]]
        Idx_known += [data['Idx_known'][i]]
        N_points += [np.sum(t_idx).astype(int)]

data['x1'] = x1
data['T1'] = T1
data['y1'] = y1
data['N_points'] = N_points
data['v'] = v
data['Idx_known'] = Idx_known

N = data['N']
Idx_unknown = [[i, j] for i in range(1, N+1) for j in range(i+1,N+1)]
idx = np.sum(np.char.add(np.char.add(np.array(Idx_unknown)[:,0].astype(str), ' + '), np.array(Idx_unknown)[:,1].astype(str))[:,np.newaxis] == np.char.add(np.char.add(np.array(Idx_known)[:,0].astype(str), ' + '), np.array(Idx_known)[:,1].astype(str))[np.newaxis,:], axis=1) == 0
Idx_unknown = np.array(Idx_unknown)[idx,:].tolist()
data['Idx_unknown'] = Idx_unknown
data['N_known'] = int(len(N_points))
data['N_unknown'] = int(len(Idx_unknown))

# convert integers to int datatypes. This is necessary to encode to json
convert_to_int(data)

# save data to json file
data_file = f'{path}/data.json'
with open(data_file, 'w') as f:
    json.dump(data, f)

