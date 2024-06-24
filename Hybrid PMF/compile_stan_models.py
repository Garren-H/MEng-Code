import os
import sys

# change stan tmpdir to home. Just a measure added for computations on the HPC which does not 
# like writing to /tmp. My change to something else if ran on a different server where /home is limited
old_tmp = os.environ['TMPDIR'] # save previous tmpdir
os.environ['TMPDIR'] = '/home/ghermanus/lustre' # update tmpdir

import cmdstanpy # type: ignore

os.environ['TMPDIR'] = old_tmp # change back to old_tmp

path = '/home/ghermanus/lustre/Hybrid PMF/Stan Models'
os.makedirs(path)

sys.path.insert(0, '/home/ghermanus/lustre/Hybrid PMF') # include home directory in path to call a python file

from generate_stan_model_code import generate_stan_code

vv = [True, False]
cc = [True, False]

for v in vv:
    for c in cc:
        model_code = generate_stan_code(include_clusters=c, variance_known=v)
        model_name = f'{path}/Hybrid_PMF_include_clusters_{c}_variance_known_{v}.stan'
        with open(model_name, 'w') as f:
            f.write(model_code)

        cmdstanpy.CmdStanModel(stan_file=model_name, cpp_options={'STAN_THREADS': True})
