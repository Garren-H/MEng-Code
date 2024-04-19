'''
This file is used to initialize the NRTL parameters and data-model mismatch parameters.
The stan code 'Stan Models/model1.stan' and 'Stan Models/model2.stan' are used to initialize
the variance parameters using the GP equivalent model followed by estimating the parameters
of the NRTL model (using the fixed variance).

The initializations are then stored and used in RegressionStep3.py file to obtain the posterior
samples using the initializations from this sheet

Note: The code was written to be executed on a linux cluster, hence the Regression.sh script 
      is used to call this script
'''

import numpy as np #type: ignore
import cmdstanpy #type: ignore
import os
import sys
import random
import json

# set seed
random.seed(1)

#set number of cpu's available
ncpu = 8

# get path from command line inputs
path = sys.argv[1]

# set stan number of threads environmental variables
os.environ['STAN_NUM_THREADS'] = f'{ncpu}'

# Set paths for data_file and stan model
data_file = f'{path}.json'
model1 = cmdstanpy.CmdStanModel(exe_file='Stan Models/model1')
model2 = cmdstanpy.CmdStanModel(exe_file='Stan Models/model2')

# Make path to store results
os.makedirs(path)

# Step 1: Sampling tau, v with random initializations
output_dir1 = f'{path}/Step1'
print('Step 1: Started. Sampling tau, v with random initializations')
fit1 = model1.sample(data=data_file, chains=5*ncpu, parallel_chains=ncpu, iter_warmup=1000,
                    iter_sampling=1000, max_treedepth=10, adapt_delta=0.8, refresh=100, output_dir=output_dir1)
print('Step 1: Completed')
print('Step 1: Diagnostics:')
print(fit1.diagnose())
print('Step 1: Summary:')
print(fit1.summary().iloc[:,3:])

# Step 2: Obtain p12, p21 using constant v
output_dir2 = f'{path}/Step2'

# create directories
os.makedirs(output_dir2)

# find index of max lp and chain with max lp
max_lp_all_idx = np.argmax(fit1.method_variables()['lp__'].T.flatten())

# extract v and save to data.json file
data = json.load(open(data_file, 'r'))
data['v'] = fit1.v[max_lp_all_idx]
with open(data_file, 'w') as f:
    json.dump(data, f)

# print inits to screen
print("Variance for Step 2:")
print(data['v'])

print('Step 2: Started. Sampling p12, p21 using constant v (from MAP) from step 1')
fit2 = model2.sample(data=data_file, chains=5*ncpu, parallel_chains=ncpu, iter_warmup=1000,
                    iter_sampling=1000, max_treedepth=10, adapt_delta=0.8, refresh=100, output_dir=output_dir2)
print('Step 2: Completed')

print('Step 2: Diagnostics:')
print(fit2.diagnose())
print('Step 2: Summary:')
print(fit2.summary().iloc[:,3:])

# Step 3. Sampling p12,p21,v using MAP initializations from previous steps
output_dir3 = f'{path}/Step3'
inits3 = f'{output_dir3}/inits.json'

# create directories to store inits
os.makedirs(output_dir3)

# extracting and saving inits
max_lp_all_idx = np.argmax(fit2.method_variables()['lp__'].T.flatten())
inits = {'p12_raw': fit2.p12_raw[max_lp_all_idx,:].tolist(),
         'p21_raw': fit2.p21_raw[max_lp_all_idx,:].tolist(),
         'v': data['v']}
with open(inits3, 'w') as f:
    json.dump(inits, f)

print('Inits for final sampling step saved. Inits are:')
print(inits)

print('Program done executing!!')
print('Exiting')
