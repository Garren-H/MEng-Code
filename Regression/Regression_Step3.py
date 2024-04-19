'''
This file is the continuation of the Regression.sh script.
It uses the stored initializations from thr Regression.sh 
script to obtain the posterior densities for a given data set
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

# Data file
data_file = f'Data/{path}.json'

# Selecting model
model3 = cmdstanpy.CmdStanModel(exe_file='Stan Models/model3')

# output directory and inits file
output_dir3 = f'Results/{path}/Step3'
inits3 = f'{output_dir3}/inits.json'

print('Step 3: Started. Sampling p12, p21, v using initializations from step 1&2')
fit3 = model3.sample(data=data_file, chains=ncpu, parallel_chains=ncpu, iter_warmup=10000,
                    iter_sampling=10000, max_treedepth=14, adapt_delta=0.99, refresh=100, 
                    thin=5, inits=inits3, output_dir=output_dir3)
print('Step 3: Completed')

print('Step 3: Diagnostics')
print(fit3.diagnose())
print('Step 3: Summary')
print(fit3.summary().iloc[:, 3:])

print('Program done executing!!')
print('Exiting')
