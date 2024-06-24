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

# Append path to obtain other functions
sys.path.append('/home/22796002')

from generate_stan_model_code import generate_stan_code # type: ignore

cc = [True, False]
vv = [True, False]
vv_MC = [True, False]

path = '/home/22796002/Pure RK PMF/Stan Models'

if not os.path.exists(path):
    os.makedirs(path)

for c in cc:
    for v in vv:
        for v_MC in vv_MC:
            stan_file = f'{path}/Pure_PMF_Include_clusters_{c}_Variance_known_{v}_Variance_MC_known_{v_MC}.stan'
            model_code = generate_stan_code(c, v, v_MC)
            with open(stan_file, 'w') as file:
                file.write(model_code)
            cmdstanpy.CmdStanModel(stan_file=stan_file, cpp_options={'STAN_THREADS': True})
