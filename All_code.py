import numpy as np #type: ignore 
import cmdstanpy #type:ignore
import os
import json
import pandas as pd #type:ignore
import shutil
from multiprocessing import Pool
import random

random.seed(1)

# class instance for shared variables
class SharedVariables:
    @classmethod
    def set_variables(cls, functional_groups, scaling = np.array([1e1, 1e-5, 1e3, 1e1]), iter_warmup=1000, iter_samples=1000,
                 chains = 4, parallel_chains=4, max_treedepth=10, adapt_delta=0.99, seed=random.randint(1, 2**32-1), a=0.3):
        cls.functional_groups=functional_groups
        cls.scaling=scaling
        cls.iter_warmup=iter_warmup
        cls.iter_samples=iter_samples
        cls.chains=chains
        cls.parallel_chains=parallel_chains
        cls.max_treedepth=max_treedepth
        cls.adapt_delta=adapt_delta
        cls.seed=seed
        cls.a=a


class subsets():
    # Python module gets the subsets data for a given subset of functional groups
    def __init__(self, functional_groups):
        self.functional_groups = functional_groups

    def get_IUPAC(self):
        with pd.ExcelFile("/home/garren/HPC Files/All Data.xlsx") as f:
            comp_names = pd.read_excel(f, sheet_name='Pure compounds')
            self.functional_groups = np.sort(self.functional_groups)
        if self.functional_groups[0] == 'all':
            # if all return all functional groups
            IUPAC_names = comp_names['IUPAC'].to_numpy()
        else:
            # else select only neccary functional groups
            idx_name = (comp_names['Functional Group'].to_numpy()[:,np.newaxis] == np.array(self.functional_groups))
            IUPAC_names = np.concatenate([comp_names['IUPAC'][idx_name[:,i]] for i in range(idx_name.shape[1])])

        return IUPAC_names

    def get_subset_df(self):
        # Get the IUPAC names based on the functional groups
        IUPAC_names = self.get_IUPAC()
        # read data
        with pd.ExcelFile("/home/garren/HPC Files/Sorted Data.xlsx") as f:
            sorted_df = pd.read_excel(f, sheet_name='Data') # Data
            Indices = pd.read_excel(f, sheet_name='Indices') # start and stop indices for each datasets nand temperature
            Indices_T = pd.read_excel(f, sheet_name='Indices_T') # start and stop indices with all temperatures
            T_index = pd.read_excel(f, sheet_name='Temperature - Index') # temperature index
        # convert to numpy -> easier to handle
        Indices = Indices.to_numpy()
        Indices_T = Indices_T.to_numpy()

        # Create a boolean to extract only relevant functional groups
        comp1_index = sorted_df['Component 1'].to_numpy()[:,np.newaxis] == IUPAC_names
        comp2_index = sorted_df['Component 2'].to_numpy()[:,np.newaxis] == IUPAC_names

        # Combines the 2 above booleans into a single boolean for processing
        comp_index = ( np.sum(comp1_index.astype(int), axis=1) + np.sum(comp2_index.astype(int),axis=1) ) == 2

        # extract only the relevant compounds
        subset_df = sorted_df[comp_index]

        # adjust the indices based on the relevant compounds
        all_indices = subset_df.index.to_numpy()
        subset_Indices = np.vstack([np.isin(all_indices, Indices[:,0]), np.isin(all_indices, Indices[:,1])]).T
        subset_Indices_T = np.vstack([np.isin(all_indices, Indices_T[:,0]), np.isin(all_indices, Indices_T[:,1])]).T
        
        # get indices for which we can get init from NRTL regression
        # That is if we have 300 datasets overall, only return 2, 3, 79, 85 if those indices have the subsets of interest
        # Convenient when storing all data
        ss = np.vstack([all_indices[subset_Indices_T[:,0]], all_indices[subset_Indices_T[:,1]]]).T
        init_indices_T = np.where(np.isin(np.char.add(Indices_T[:,0].astype(str), Indices_T[:,1].astype(str)), np.char.add(ss[:,0].astype(str), ss[:,1].astype(str))))[0]

        ss = np.vstack([all_indices[subset_Indices[:,0]], all_indices[subset_Indices[:,1]]]).T
        init_indices = np.where(np.isin(np.char.add(Indices[:,0].astype(str), Indices[:,1].astype(str)), np.char.add(ss[:,0].astype(str), ss[:,1].astype(str))))[0]
        
        # Adjust the start and stop indices
        subset_Indices = np.where(subset_Indices)[0].reshape((-1,2))
        subset_Indices_T = np.where(subset_Indices_T)[0].reshape((-1,2))
        
        # reset the dataframe indecing
        subset_df = subset_df.reset_index(drop=True)

        # Remove IUPAC for which we do not have experimental data in the subset
        #subset_comp1_index = IUPAC_names[:,np.newaxis] == subset_df['Component 1'][subset_Indices_T[:,0]].to_numpy()
        #subset_comp2_index = IUPAC_names[:,np.newaxis] == subset_df['Component 2'][subset_Indices_T[:,0]].to_numpy()
        #idx_keep = np.sum(subset_comp1_index, axis=1) + np.sum(subset_comp2_index, axis=1) != 0
        #IUPAC_names = IUPAC_names[idx_keep]

        # Store compound and temperature information in a dict for convenience
        Info_Indices = {'Component names': {'IUPAC': IUPAC_names,
                                        'Index': np.arange(len(IUPAC_names))},
                    'Temperature': {'Temperature [K]': T_index['Temperature [K]'][np.isin(T_index['Index'].to_numpy(), np.unique(subset_df['Temperature - Index'].to_numpy()) )].to_numpy(),
                                    'Index': np.arange(len(np.unique(subset_df['Temperature - Index'].to_numpy())))}}    

        # Check if subset df has for all the compounds. I.e. there is no fully missing rows or columns

        # Adjust Component and temperature indices
        subset_df['Component 1 - Index'] = ((subset_df['Component 1'].to_numpy()[:, np.newaxis] == Info_Indices['Component names']['IUPAC']).astype(int)@np.arange(Info_Indices['Component names']['IUPAC'].shape[0])[:, np.newaxis]).flatten()
        subset_df['Component 2 - Index'] = ((subset_df['Component 2'].to_numpy()[:, np.newaxis] == Info_Indices['Component names']['IUPAC']).astype(int)@np.arange(Info_Indices['Component names']['IUPAC'].shape[0])[:, np.newaxis]).flatten()
        subset_df['Temperature - Index'] = ((subset_df['Temperature - Index'].to_numpy()[:, np.newaxis] == np.unique(subset_df['Temperature - Index'].to_numpy())).astype(int)@np.arange(np.unique(subset_df['Temperature - Index'].to_numpy()).shape[0])[:, np.newaxis]).flatten()

        # return all the relevant data
        return subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T
    
    def __del__(self):
        del self

def NRTL_MAP(model, data_file, output_dir, n):
    seed = random.randint(1, 2**32-1)
    counter = 0
    e = True
    while counter<=5000 and e:
        try:
            fit_no_var = model.optimize(data=data_file, algorithm='lbfgs',
                                        refresh=1, jacobian=True, tol_rel_grad=1,
                                        iter=5000, output_dir=f'{output_dir}/{n}',
                                        seed=seed)
            e = False
        except:
            counter += 1
            seed = random.randint(1, 2**32-1)
            shutil.rmtree(f'{output_dir}/{n}')
    return fit_no_var

class NRTL_regression():
    def __init__(self, T_dependent=True, constant_alpha=True):
        self.T_dependent = T_dependent
        self.constant_alpha=constant_alpha
        self.iter_warmup=SharedVariables.iter_warmup
        self.iter_samples=SharedVariables.iter_samples
        self.chains = SharedVariables.chains
        self.parallel_chains = SharedVariables.parallel_chains
        self.max_treedepth = SharedVariables.max_treedepth
        self.adapt_delta = SharedVariables.adapt_delta
        self.scaling = SharedVariables.scaling
        self.seed = SharedVariables.seed
        self.a=SharedVariables.a
        self.folder = 'Regression'
        self.N = pd.read_excel('Sorted Data.xlsx', sheet_name='Indices_T').to_numpy().shape[0]
    
    def store_data(self):
        print('Started: Storing data')
        # save the data to json files for processing in cmdstanpy
        sub_folder = 'Data'
        # if folder does not exists create the folder
    
        # Get the relevant data from
        subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T = subsets(np.array(['all'])).get_subset_df()
        
        # Get the number of datasets
        self.N = subset_Indices_T.shape[0]
        # Loop through datasets and create dirs
        for i in range(self.N):
            if not os.path.exists(f'{self.folder}/{sub_folder}/T_dependent_True'):
                os.makedirs(f'{self.folder}/{sub_folder}/T_dependent_True')
            # create dictionary to save values
            data = {'Component 1': str(subset_df['Component 1'][subset_Indices_T[i,0]]),
                    'Component 2': str(subset_df['Component 2'][subset_Indices_T[i,0]]),
                    'N_points': int(subset_Indices_T[i,1] + 1 - subset_Indices_T[i,0]),
                    'x': subset_df['Composition component 1 [mol/mol]'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].tolist(),
                    'T': subset_df['Temperature [K]'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].tolist(),
                    'y': subset_df['Excess Enthalpy [J/mol]'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].tolist(),
                    'T_index': subset_df['Temperature - Index'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[i,1]+1].tolist(),
                    'scaling': self.scaling.tolist(),
                    'a': self.a}
            with open(f'{self.folder}/{sub_folder}/T_dependent_True/{i}.json', 'w') as f:
                json.dump(data,f)

            unique_T = np.unique(np.array(data['T_index']))
            for j in range(len(unique_T)):
                if not os.path.exists(f'{self.folder}/{sub_folder}/T_dependent_False'):
                    os.makedirs(f'{self.folder}/{sub_folder}/T_dependent_False')
                idx = np.array(data['T_index']) == j
                new_data = {'Component 1': data['Component 1'],
                            'Component 2': data['Component 2'],
                            'N_points': int(np.sum(idx)),
                            'x': (np.array(data['x'])[idx]).tolist(),
                            'T': (np.array(data['T'])[idx]).tolist(),
                            'y': (np.array(data['y'])[idx]).tolist(),
                            'scaling': self.scaling.tolist(),
                            'a': self.a}
                with open(f'{self.folder}/{sub_folder}/T_dependent_False/{i}_{j}.json', 'w') as f:
                    json.dump(data,f)

        print('Completed: Storing data\n\n')
    
    def create_stan_models(self):
        print('Started: creating Stan models')
        # check if files exist
        sub_folder = 'Stan Models/lin_inv_ln'
        if not os.path.exists(f'{self.folder}/{sub_folder}'):
            os.makedirs(f'{self.folder}/{sub_folder}')

        constant_alpha_no_var = '''
            functions {
                vector NRTL(vector x, vector T, vector p12, vector p21, real a, vector scaling) {
                    int N = rows(x);
                    vector[N] t12 = p12[1]*scaling[1] + p12[2]*scaling[2] * T + p12[3]*scaling[3] ./T + p12[4]*scaling[4] * log(T);
                    vector[N] t21 = p21[1]*scaling[1] + p21[2]*scaling[2] * T + p21[3]*scaling[3] ./T + p21[4]*scaling[4] * log(T);
                    vector[N] G12 = exp(-a * t12);
                    vector[N] G21 = exp(-a * t21);
                    vector[N] dt12_dT = p12[2]*scaling[2] - p12[3]*scaling[3] ./ square(T) + p12[4]*scaling[4] ./ T;
                    vector[N] dt21_dT = p21[2]*scaling[2] - p21[3]*scaling[3] ./ square(T) + p21[4]*scaling[4] ./ T;
                    vector[N] term1 = ( ( (1-x) .* G12 .* (1 - a*t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
                    vector[N] term2 = ( ( x .* G21 .* (1 - a*t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
                    return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
                }
            }

            data {
                int N_points;
                vector<lower=0, upper=1>[N_points] x;
                vector[N_points] y;
                vector<lower=0>[N_points] T;
                vector[4] scaling;
                real<lower=0> a;
            }

            transformed data {
                real<lower=0> error=0.01;
            }

            parameters {
                vector[4] p12;
                vector[4] p21;
            }

            model {
                p12 ~ normal(0,5);
                p21 ~ normal(0,5);
                {
                    vector[N_points] y_means = NRTL(x, T, p12, p21, a, scaling);
                    y ~ normal(y_means, error*abs(y)+1e-3);
                }
            }
        '''

        with open(f'{self.folder}/{sub_folder}/constant_alpha_no_var.stan', 'w') as f:
            f.write(constant_alpha_no_var)

        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/constant_alpha_no_var.stan')

        constant_alpha_var = '''
            functions {
                vector NRTL(vector x, vector T, vector p12, vector p21, real a, vector scaling) {
                    int N = rows(x);
                    vector[N] t12 = p12[1]*scaling[1] + p12[2]*scaling[2] * T + p12[3]*scaling[3] ./T + p12[4]*scaling[4] * log(T);
                    vector[N] t21 = p21[1]*scaling[1] + p21[2]*scaling[2] * T + p21[3]*scaling[3] ./T + p21[4]*scaling[4] * log(T);
                    vector[N] G12 = exp(-a * t12);
                    vector[N] G21 = exp(-a * t21);
                    vector[N] dt12_dT = p12[2]*scaling[2] - p12[3]*scaling[3] ./ square(T) + p12[4]*scaling[4] ./ T;
                    vector[N] dt21_dT = p21[2]*scaling[2] - p21[3]*scaling[3] ./ square(T) + p21[4]*scaling[4] ./ T;
                    vector[N] term1 = ( ( (1-x) .* G12 .* (1 - a*t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
                    vector[N] term2 = ( ( x .* G21 .* (1 - a*t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
                    return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
                }
            }

            data {
                int N_points;
                vector<lower=0, upper=1>[N_points] x;
                vector[N_points] y;
                vector<lower=0>[N_points] T;
                vector[4] scaling;
                real<lower=0> a;
            }

            transformed data {
                real<lower=0> error=0.01;
            }

            parameters {
                vector[4] p12;
                vector[4] p21;
                real<lower=0> v;
            }

            model {
                v ~ inv_gamma(2,1);
                p12 ~ normal(0,5);
                p21 ~ normal(0,5);
                {
                    vector[N_points] y_means = NRTL(x, T, p12, p21, a, scaling);
                    y ~ normal(y_means, error*abs(y)+sqrt(v));
                }
            }
        '''

        with open(f'{self.folder}/{sub_folder}/constant_alpha_var.stan', 'w') as f:
            f.write(constant_alpha_var)

        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/constant_alpha_var.stan')

        var_alpha_no_var = '''
            functions {
                vector NRTL(vector x, vector T, vector p12, vector p21, real a, vector scaling) {
                    int N = rows(x);
                    vector[N] t12 = p12[1]*scaling[1] + p12[2]*scaling[2] * T + p12[3]*scaling[3] ./T + p12[4]*scaling[4] * log(T);
                    vector[N] t21 = p21[1]*scaling[1] + p21[2]*scaling[2] * T + p21[3]*scaling[3] ./T + p21[4]*scaling[4] * log(T);
                    vector[N] G12 = exp(-a * t12);
                    vector[N] G21 = exp(-a * t21);
                    vector[N] dt12_dT = p12[2]*scaling[2] - p12[3]*scaling[3] ./ square(T) + p12[4]*scaling[4] ./ T;
                    vector[N] dt21_dT = p21[2]*scaling[2] - p21[3]*scaling[3] ./ square(T) + p21[4]*scaling[4] ./ T;
                    vector[N] term1 = ( ( (1-x) .* G12 .* (1 - a*t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
                    vector[N] term2 = ( ( x .* G21 .* (1 - a*t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
                    return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
                }
            }

            data {
                int N_points;
                vector<lower=0, upper=1>[N_points] x;
                vector[N_points] y;
                vector<lower=0>[N_points] T;
                vector[4] scaling;
            }

            transformed data {
                real<lower=0> error=0.01;
            }

            parameters {
                vector[4] p12;
                vector[4] p21;
                real b;
            }

            transformed parameters {
                real<lower=0.1, upper=0.5> a = 0.4/(1 + exp(b)) + 0.1; 
            }

            model {
                p12 ~ normal(0,5);
                p21 ~ normal(0,5);
                b ~ normal(0,1);
                {
                    vector[N_points] y_means = NRTL(x, T, p12, p21, a, scaling);
                    y ~ normal(y_means, error*abs(y)+1e-3);
                }
                target += abs( -0.4/( (0.5-a)*(a-0.1) ) );
            }
        '''

        with open(f'{self.folder}/{sub_folder}/var_alpha_no_var.stan', 'w') as f:
            f.write(var_alpha_no_var)

        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/var_alpha_no_var.stan')

        var_alpha_var = '''
            functions {
                vector NRTL(vector x, vector T, vector p12, vector p21, real a, vector scaling) {
                    int N = rows(x);
                    vector[N] t12 = p12[1]*scaling[1] + p12[2]*scaling[2] * T + p12[3]*scaling[3] ./T + p12[4]*scaling[4] * log(T);
                    vector[N] t21 = p21[1]*scaling[1] + p21[2]*scaling[2] * T + p21[3]*scaling[3] ./T + p21[4]*scaling[4] * log(T);
                    vector[N] G12 = exp(-a * t12);
                    vector[N] G21 = exp(-a * t21);
                    vector[N] dt12_dT = p12[2]*scaling[2] - p12[3]*scaling[3] ./ square(T) + p12[4]*scaling[4] ./ T;
                    vector[N] dt21_dT = p21[2]*scaling[2] - p21[3]*scaling[3] ./ square(T) + p21[4]*scaling[4] ./ T;
                    vector[N] term1 = ( ( (1-x) .* G12 .* (1 - a*t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
                    vector[N] term2 = ( ( x .* G21 .* (1 - a*t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
                    return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
                }
            }

            data {
                int N_points;
                vector<lower=0, upper=1>[N_points] x;
                vector[N_points] y;
                vector<lower=0>[N_points] T;
                vector[4] scaling;
            }

            transformed data {
                real<lower=0> error=0.01;
            }

            parameters {
                vector[4] p12;
                vector[4] p21;
                real b;
                real<lower=0> v;
            }

            transformed parameters {
                real<lower=0.1, upper=0.5> a = 0.4/(1 + exp(b)) + 0.1; 
            }

            model {
                b ~ normal(0,1);
                v ~ inv_gamma(2,1);
                p12 ~ normal(0,5);
                p21 ~ normal(0,5);
                {
                    vector[N_points] y_means = NRTL(x, T, p12, p21, a, scaling);
                    y ~ normal(y_means, error*abs(y)+sqrt(v));
                }
                target += abs( -0.4/( (0.5-a)*(a-0.1) ) );
            }
        '''

        with open(f'{self.folder}/{sub_folder}/var_alpha_var.stan', 'w') as f:
            f.write(var_alpha_var)

        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/var_alpha_var.stan')

        print('Completed: creating Stan models\n\n')


    def chose_stan_model(self):
        print('Started: Chosing stan models')
        sub_folder = 'Stan Models/lin_inv_ln'
        if self.constant_alpha:
            self.model_var = cmdstanpy.CmdStanModel(exe_file = f'{self.folder}/{sub_folder}/constant_alpha_var')
            self.model_no_var = cmdstanpy.CmdStanModel(exe_file = f'{self.folder}/{sub_folder}/constant_alpha_no_var')
        else:
            self.model_var = cmdstanpy.CmdStanModel(exe_file = f'{self.folder}/{sub_folder}/var_alpha_var')
            self.model_no_var = cmdstanpy.CmdStanModel(exe_file = f'{self.folder}/{sub_folder}/var_alpha_no_var')

        print('Completed: Chosing stan models\n\n')

    def Regression(self):
        print('Started: Performing regression')
        # Do regression based for all datasets for constant alpha, variable alpha and 
        sub_folder = f'Results/lin_inv_ln/T_dependent_{self.T_dependent}/Constant_alpha_{self.constant_alpha}'
        if self.T_dependent:
            for i in range(self.N):
                print(f'Data {i+1} out of {self.N}')
                with Pool(6) as pool:
                    fit_no_var = pool.starmap(NRTL_MAP, [[self.model_no_var, f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}.json',
                                                                    f'{self.folder}/{sub_folder}/MAP/No_var/{i}', n] for n in range(6)])
                    fit_no_var += pool.starmap(NRTL_MAP, [[self.model_no_var, f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}.json',
                                                                    f'{self.folder}/{sub_folder}/MAP/No_var/{i}', n] for n in range(6, 12)])
                    fit_no_var += pool.starmap(NRTL_MAP, [[self.model_no_var, f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}.json',
                                                                    f'{self.folder}/{sub_folder}/MAP/No_var/{i}', n] for n in range(12,18)])

                max_lp = np.argmax(np.array([fit_no_var[n].optimized_params_dict['lp__'] for n in range(18)]))

                inits = {'p12': fit_no_var[max_lp].p12,
                        'p21': fit_no_var[max_lp].p21}
                try: 
                    inits['b'] = fit_no_var[max_lp].b
                except:
                    pass
                
                inits['v'] = 0.5
                
                self.model_var.sample(data=f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}.json', inits=inits, chains=self.chains,
                                    parallel_chains=self.parallel_chains, 
                                    iter_warmup=self.iter_warmup, iter_sampling=self.iter_samples,
                                    max_treedepth=self.max_treedepth, adapt_delta=self.adapt_delta, 
                                    output_dir=f'{self.folder}/{sub_folder}/Sampling/Var/{i}')
        
        else:
            for i in range(self.N):
                with open(f'{self.folder}/Data/T_dependent_True/{i}.json', 'r') as f:
                    data = json.load(f)
                    num_T = len(np.unique(np.array(data['T_index'])))
                    del data
                for j in range(num_T):
                    with Pool(6) as pool:
                        fit_no_var = pool.starmap(NRTL_MAP, [[self.model_no_var, f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}_{j}.json', 
                                                             f'{self.folder}/{sub_folder}/MAP/No_var/{i}/{j}', n] for n in range(6)])
                        fit_no_var += pool.starmap(NRTL_MAP, [[self.model_no_var, f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}_{j}.json', 
                                                             f'{self.folder}/{sub_folder}/MAP/No_var/{i}/{j}', n] for n in range(6, 12)])
                        fit_no_var += pool.starmap(NRTL_MAP, [[self.model_no_var, f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}_{j}.json', 
                                                             f'{self.folder}/{sub_folder}/MAP/No_var/{i}/{j}', n] for n in range(12, 18)])
                    
                    max_lp = np.argmax(np.array([fit_no_var[n].optimized_params_dict['lp__'] for n in range(18)]))

                    inits = {'p12': fit_no_var[max_lp].p12,
                            'p21': fit_no_var[max_lp].p21}
                    try: 
                        inits['b'] = fit_no_var[max_lp].b
                    except:
                        pass
                    
                    inits['v'] = 0.5

                    self.model_var.sample(data=f'{self.folder}/Data/T_dependent_{self.T_dependent}/{i}_{j}.json', inits=inits, chains=self.chains,
                                        parallel_chains=self.parallel_chains, 
                                        iter_warmup=self.iter_warmup, iter_sampling=self.iter_samples,
                                        max_treedepth=self.max_treedepth, adapt_delta=self.adapt_delta, 
                                        output_dir=f'{self.folder}/{sub_folder}/Sampling/Var/{i}/{j}')
        print('Completed: Performing regression')
        
    def __del__(self):
        del self

class PMF():
    def __init__(self, D, T_dependent=True, constant_alpha=True):
        self.D = D
        self.functional_groups = SharedVariables.functional_groups
        self.T_dependent = T_dependent
        self.constant_alpha=constant_alpha
        self.iter_warmup=SharedVariables.iter_warmup
        self.iter_samples=SharedVariables.iter_samples
        self.chains = SharedVariables.chains
        self.parallel_chains = SharedVariables.parallel_chains
        self.max_treedepth = SharedVariables.max_treedepth
        self.adapt_delta = SharedVariables.adapt_delta
        self.scaling = SharedVariables.scaling
        self.seed = SharedVariables.seed
        self.a=SharedVariables.a
        self.folder = 'PMF'
        self.subset = ''
        for func in self.functional_groups:
            if self.subset == '':
                self.subset = func
            else:
                self.subset += '_'+func

    def get_indices_and_data(self):
        try:
            subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, self.init_indices_T = subsets(self.functional_groups).get_subset_df()
        except:
            print('Invalid functional groups supplied')
        self.N_datasets = subset_Indices_T.shape[0]
        if self.T_dependent:
            # observed indice [i,j] for the subset of interest
            self.Indices_obs = np.vstack([subset_df['Component 1 - Index'][subset_Indices_T[:,0]], subset_df['Component 2 - Index'][subset_Indices_T[:,0]]]).T

        else:
            # observed indices [i,j,T] for the subset of interest
            self.Indices_obs = np.vstack([subset_df['Component 1 - Index'][subset_Indices[:,0]], subset_df['Component 2 - Index'][subset_Indices[:,0]], subset_df['Temperature - Index'][subset_Indices[:,0]]]).T
        
        sub_folder = 'Data'
        sub_sub_folder = 'lin_inv_lin'

        self.N_points = (subset_Indices_T[:,1] + 1 - subset_Indices_T[:,0]).astype(int)
        self.N_max_points = np.max(self.N_max_points)

        self.data_sampling = {'N_datasets': int(self.N_datasets),
                'N_max_points': int(self.N_max_points),
                'N_points': self.N_points.astype(int).tolist(),
                'Indices_obs': (self.Indices_obs+1).astype(int).tolist(),
                'N': int(np.max(Info_Indices['Compounds']['Index'])+1),
                'x': np.zeros((self.N_datasets, self.N_max_points)),
                'T': 298.15*np.ones((self.N_datasets, self.N_max_points)),
                'y': np.zeros((self.N_datasets, self.N_max_points)),
                'D': int(self.D),
                'scaling': self.scaling.tolist(),
                'a': self.a}
        path_to_regression_sampling = f'Regression/Results/lin_inv_ln/T_dependent_{self.T_dependent}/Constant_alpha_{self.constant_alpha}'
        regression_csv_files = [[f'{path_to_regression_sampling}/{i}/{f}' for f in os.listdir(f'{path_to_regression_sampling}/{i}') if f.endswith('.csv')] for i in self.init_indices_T]
        p12_mean = np.zeros((len(regression_csv_files), 4))
        p21_mean = p12_mean.copy()
        p12_std = p12_mean.copy()
        p21_std = p12_mean.copy()
        b_mean = p12_mean.copy()
        b_std = p12_mean.copy()
        for i in range(len(regression_csv_files)):
            fit = cmdstanpy.from_csv(regression_csv_files[i])
            p12_mean[i,:] = np.mean(fit.p12, axis=0)
            p21_mean[i,:] = np.mean(fit.p21, axis=0)
            p12_std[i,:] = np.std(fit.p12, axis=0)
            p21_std[i,:] = np.std(fit.p21, axis=0)
            try: 
                b_mean[i] = np.mean(fit.b)
                b_std[i] = np.std(fit.b)
            except:
                remove_b = True
            del fit
        if remove_b:
            del b
        self.data_MAP = [{'a_obs': np.vstack([p12_mean[:,0], p21_mean[:,0]]).flatten(),
                     'a_v': np.vstack([p12_std[:,0], p21_std[:,0]]).flatten()},
                    {'a_obs': np.vstack([p12_mean[:,1], p21_mean[:,1]]).flatten(),
                     'a_v': np.vstack([p12_std[:,1], p21_std[:,1]]).flatten()},
                    {'a_obs': np.vstack([p12_mean[:,2], p21_mean[:,2]]).flatten(),
                     'a_v': np.vstack([p12_std[:,2], p21_std[:,2]]).flatten()},
                    {'a_obs': np.vstack([p12_mean[:,3], p21_mean[:,3]]).flatten(),
                     'a_v': np.vstack([p12_std[:,3], p21_std[:,3]]).flatten()},
                    ]
        if not remove_b:
            self.data_MAP.append({'a_obs': b_mean,
                             'a_v': b_std})
        self.data_MAP_needed = {'N': self.data_sampling['N'],
                           'D': self.D,
                           'N_obs': 2*self.N_datasets,
                           'Indices_obs': np.column_stack([self.Indices_obs, self.Indices_obs[:,-1::]])}
        
        for i in range(self.N_datasets):
            self.data_sampling['x'][i,:self.N_points[i]] = subset_df['Composition component 1 [mol/mol]'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[:,1]+1]
            self.data_sampling['T'][i,:self.N_points[i]] = subset_df['Temperature [K]'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[:,1]+1]
            self.data_sampling['y'][i,:self.N_points[i]] = subset_df['Excess Enthalpy [J/mol]'].to_numpy()[subset_Indices_T[i,0]:subset_Indices_T[:,1]+1]
        
        
    def create_stan_models(self):
        sub_folder = 'Stan models'
        if not os.path.exists(f'{self.folder}/{sub_folder}'):
            os.makedirs(f'{self.folder}/{sub_folder}')
        MAP = '''

        data {
            int N_obs;
            int N;
            int D;
            vector[N_obs] a_obs;
            vector[N_obs] a_v;
            array[N_obs,2] int Indices_obs;
            real<lower=0> v_zeros;
        }

        transformed data {
            real error = 0.01;
        }

        parameters {
            matrix[D,N] U_raw;
            matrix[D,N] V_raw;
            vector<lower=0>[D] v_D;
            real<lower=0> scale;
        }

        model {
            matrix[D,N] U;
            matrix[D,N] V;
            matrix[N,N] A;
            vector[N_obs] a_means;

            scale ~ gamma(3,2);
            v_D ~ exponential(scale);
            for (i in 1:D) {
                U_raw[i,:] ~ std_normal();
                V_raw[i,:] ~ std_normal();
            }

            U = U_raw .* rep_matrix(sqrt(v_D),N);
            V = U_raw .* rep_matrix(sqrt(v_D),N);
            A = U' * V;

            for (i in 1:N_obs) {
                a_means[i] = A[Indices_obs[i,1], Indices_obs[i,2]];
            }

            a_obs ~ normal(a_means, a_v);
            0 ~ normal(diagonal(A), v_zeros);
        }
        '''

        with open(f'{self.folder}/{sub_folder}/MAP.stan', 'w') as f:
            f.write(MAP)

        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/MAP.stan')

        sampling_constant_alpha = '''
        functions {
            matrix NRTL(matrix x, matrix T, matrix p12, matrix p21, real a, vector scaling) {
                int N = rows(x);
                int M = rows(x');
                matrix[N,M] t12 = rep_matrix(p12[:,1]*scaling[1], M) + rep_matrix(p12[:,2]*scaling[2], M) .* T + rep_matrix(p12[:,3]*scaling[3], M) ./ T + rep_matrix(p12[:,4]*scaling[4], M) .* log(T);
                matrix[N,M] t21 = rep_matrix(p21[:,1]*scaling[1], M) + rep_matrix(p21[:,2]*scaling[2], M) .* T + rep_matrix(p21[:,3]*scaling[3], M) ./ T + rep_matrix(p21[:,4]*scaling[4], M) .* log(T);
                matrix[N,M] G12 = exp(-a .* t12);
                matrix[N,M] G21 = exp(-a .* t21);
                matrix[N,M] dt12_dT = rep_matrix(p12[:,2]*scaling[2], M) - rep_matrix(p12[:,3]*scaling[3], M) ./ square(T) + rep_matrix(p12[:,4]*scaling[4], M) ./ T;
                matrix[N,M] dt21_dT = rep_matrix(p21[:,2]*scaling[2], M) - rep_matrix(p21[:,3]*scaling[3], M) ./ square(T) + rep_matrix(p21[:,4]*scaling[4], M) ./ T;
                matrix[N,M] term1 = ( ( (1-x) .* G12 .* (1 - a .* t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
                matrix[N,M] term2 = ( ( x .* G21 .* (1 - a .* t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
                return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
            }
        }

        data {
            int N_datasets;
            int N_max_points;
            array[N_datasets] int N_points;
            matrix[N_datasets, N_max_points] x;
            matrix[N_datasets, N_max_points] y;
            matrix[N_datasets, N_max_points] T;
            array[N_datasets, 2] int Indices_obs;
            int D;
            int N;
            real a;
            vector[4] scaling;
        }

        transformed data {
            real error = 0.01;
        }

        parameters {
            // PMF parameters
            array[8] matrix[D, N] F_raw; // array of feature matrices [U,V]
            real<lower=0> scale; // scale parameter for exponential ditribution
            vector<lower=0>[D] Fv; // variance of the i-th row

            // data parameters
            vector<lower=0>[N_datasets] v; //variance of the i-th dataset 
        }

        model {
            matrix[N_datasets,4] p12;
            matrix[N_datasets,4] p21;
            matrix[N,N] A_p1;
            matrix[N,N] A_p2;
            matrix[N,N] A_p3;
            matrix[N,N] A_p4;
            array[8] matrix[D,N] F;

            // PMF priors
            scale ~ gamma(3,2);
            Fv ~ exponential(scale); // expoential prios on varaince of the latent features
            for (i in 1:D) {
                F_raw[1,i,:] ~ std_normal(); // normal prior on latent features
                F_raw[2,i,:] ~ std_normal();
                F_raw[3,i,:] ~ std_normal();
                F_raw[4,i,:] ~ std_normal();
                F_raw[5,i,:] ~ std_normal();
                F_raw[6,i,:] ~ std_normal();
                F_raw[7,i,:] ~ std_normal();
                F_raw[8,i,:] ~ std_normal();
            }

            // scale with variance
            for (i in 1:8) {
                F[i] = F_raw[i] .* rep_matrix(sqrt(Fv), N);
            }

            // matrices
            A_p1 = (F[1,:,:]' * F[2,:,:]);
            A_p2 = (F[3,:,:]' * F[4,:,:]);
            A_p3 = (F[5,:,:]' * F[6,:,:]);
            A_p4 = (F[7,:,:]' * F[8,:,:]);

            for (i in 1:N_datasets) {
                p12[i,1] = A_p1[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,1] = A_p1[Indices_obs[i,2], Indices_obs[i,1]];
                p12[i,2] = A_p2[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,2] = A_p2[Indices_obs[i,2], Indices_obs[i,1]];
                p12[i,3] = A_p3[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,3] = A_p3[Indices_obs[i,2], Indices_obs[i,1]];
                p12[i,4] = A_p4[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,4] = A_p4[Indices_obs[i,2], Indices_obs[i,1]]; 
            }

            // Zero's on diagonal 
            0 ~ normal(diagonal(A_p1), 1e-3);
            0 ~ normal(diagonal(A_p2), 1e-3);
            0 ~ normal(diagonal(A_p3), 1e-3);
            0 ~ normal(diagonal(A_p4), 1e-3);

            // data priors
            v ~ inv_gamma(2,1);
            
            //likelihood
            {
                matrix[N_datasets, N_max_points] y_means = NRTL(x, T, p12, p21, a, scaling);
                for ( i in 1:N_datasets) {
                    y[i,1:N_points[i]]' ~ normal(y_means[i,1:N_points[i]]', error*abs(y_means[i,1:N_points[i]]')+sqrt(v[i]));
                }
            } 
        }
        '''

        with open(f'{self.folder}/{sub_folder}/sampling_constant_alpha.stan', 'w') as f:
            f.write(sampling_constant_alpha)
        
        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/sampling_constant_alpha.stan')

        sampling_var_alpha = '''
        functions {
            matrix NRTL(matrix x, matrix T, matrix p12, matrix p21, vector a, vector scaling) {
                int N = rows(x);
                int M = rows(x');
                matrix[M,N] alpha = rep_matrix(a, M);
                matrix[N,M] t12 = rep_matrix(p12[:,1]*scaling[1], M) + rep_matrix(p12[:,2]*scaling[2], M) .* T + rep_matrix(p12[:,3]*scaling[3], M) ./ T + rep_matrix(p12[:,4]*scaling[4], M) .* log(T);
                matrix[N,M] t21 = rep_matrix(p21[:,1]*scaling[1], M) + rep_matrix(p21[:,2]*scaling[2], M) .* T + rep_matrix(p21[:,3]*scaling[3], M) ./ T + rep_matrix(p21[:,4]*scaling[4], M) .* log(T);
                matrix[N,M] G12 = exp(-alpha .* t12);
                matrix[N,M] G21 = exp(-alpha .* t21);
                matrix[N,M] dt12_dT = rep_matrix(p12[:,2]*scaling[2], M) - rep_matrix(p12[:,3]*scaling[3], M) ./ square(T) + rep_matrix(p12[:,4]*scaling[4], M) ./ T;
                matrix[N,M] dt21_dT = rep_matrix(p21[:,2]*scaling[2], M) - rep_matrix(p21[:,3]*scaling[3], M) ./ square(T) + rep_matrix(p21[:,4]*scaling[4], M) ./ T;
                matrix[N,M] term1 = ( ( (1-x) .* G12 .* (1 - alpha .* t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
                matrix[N,M] term2 = ( ( x .* G21 .* (1 - alpha .* t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
                return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
            }
        }

        data {
            int N_datasets;
            int N_max_points;
            array[N_datasets] int N_points;
            matrix[N_datasets, N_max_points] x;
            matrix[N_datasets, N_max_points] y;
            matrix[N_datasets, N_max_points] T;
            array[N_datasets, 2] int Indices_obs;
            int D;
            int N;
            vector[4] scaling;
        }

        transformed data {
            real error = 0.01;
        }

        parameters {
            // PMF parameters
            array[9] matrix[D, N] F_raw; // array of feature matrices [U,V]
            real<lower=0> scale; // scale parameter for exponential ditribution
            vector<lower=0>[D] Fv; // variance of the i-th row

            // data parameters
            vector<lower=0>[N_datasets] v; //variance of the i-th dataset 
        }

        model {
            matrix[N_datasets,4] p12;
            matrix[N_datasets,4] p21;
            vector[N_datasets] a;
            matrix[N,N] A_p1;
            matrix[N,N] A_p2;
            matrix[N,N] A_p3;
            matrix[N,N] A_p4;
            matrix[N,N] A_b;
            array[8] matrix[D,N] F;

            // PMF priors
            scale ~ gamma(3,2);
            Fv ~ exponential(scale); // expoential prios on varaince of the latent features
            for (i in 1:D) {
                F_raw[1,i,:] ~ std_normal(); // normal prior on latent features
                F_raw[2,i,:] ~ std_normal();
                F_raw[3,i,:] ~ std_normal();
                F_raw[4,i,:] ~ std_normal();
                F_raw[5,i,:] ~ std_normal();
                F_raw[6,i,:] ~ std_normal();
                F_raw[7,i,:] ~ std_normal();
                F_raw[8,i,:] ~ std_normal();
                F_raw[9,i,:] ~ std_normal();
            }

            // scale with variance
            for (i in 1:9) {
                F[i] = F_raw[i] .* rep_matrix(sqrt(Fv), N);
            }

            // matrices
            A_p1 = (F[1,:,:]' * F[2,:,:]);
            A_p2 = (F[3,:,:]' * F[4,:,:]);
            A_p3 = (F[5,:,:]' * F[6,:,:]);
            A_p4 = (F[7,:,:]' * F[8,:,:]);
            A_b = (F[9,:,:]' * F[9,:,:]);

            for (i in 1:N_datasets) {
                p12[i,1] = A_p1[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,1] = A_p1[Indices_obs[i,2], Indices_obs[i,1]];
                p12[i,2] = A_p2[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,2] = A_p2[Indices_obs[i,2], Indices_obs[i,1]];
                p12[i,3] = A_p3[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,3] = A_p3[Indices_obs[i,2], Indices_obs[i,1]];
                p12[i,4] = A_p4[Indices_obs[i,1], Indices_obs[i,2]];
                p21[i,4] = A_p4[Indices_obs[i,2], Indices_obs[i,1]]; 
                a[i] = 0.4 ./ ( 1 + exp( A_b[Indices_obs[i,1], Indices_obs[i,2]] ) ) + 0.1;
            }

            // Zero's on diagonal 
            0 ~ normal(diagonal(A_p1), 1e-3);
            0 ~ normal(diagonal(A_p2), 1e-3);
            0 ~ normal(diagonal(A_p3), 1e-3);
            0 ~ normal(diagonal(A_p4), 1e-3);
            0 ~ normal(diagonal(A_b), 1e-3);

            // data priors
            v ~ inv_gamma(2,1);
            
            //likelihood
            {
                matrix[N_datasets, N_max_points] y_means = NRTL(x, T, p12, p21, a, scaling);
                for ( i in 1:N_datasets) {
                    y[i,1:N_points[i]]' ~ normal(y_means[i,1:N_points[i]]', error*abs(y_means[i,1:N_points[i]]')+sqrt(v[i]));
                }
            } 
        }
        '''

        with open(f'{self.folder}/{sub_folder}/sampling_var_alpha.stan', 'w') as f:
            f.write(sampling_var_alpha)
        
        cmdstanpy.CmdStanModel(stan_file=f'{self.folder}/{sub_folder}/sampling_var_alpha.stan')

    def chose_stan_model(self):
        sub_folder = 'Stan models'
        self.MAP_model = cmdstanpy.CmdStanModel(exe_file=f'{self.folder}/{sub_folder}/MAP')
        if self.constant_alpha:
            self.sampling_model = cmdstanpy.CmdStanModel(exe_file=f'{self.folder}/{sub_folder}/sampling_constant_alpha')
        else:
            self.MAP_alpha_model = cmdstanpy.CmdStanModel(exe_file=f'{self.folder}/{sub_folder}/MAP_alpha')
            self.sampling_model = cmdstanpy.CmdStanModel(exe_file=f'{self.folder}/{sub_folder}/sampling_var_alpha')

    def perform_sampling(self):
        sub_folder = f'Results/lin_inv_ln/{self.subset}/constant_alpha_{self.constant_alpha}'

        def find_MAP_one(stan_model, output_folder, data):
            seed = random.randint(1, 2**32-1)
            counter=0
            e = True
            all_data = self.data_MAP_needed
            for key, val in data.items():
                all_data[key] = val
            while counter<=2000 and e:
                try:
                    fit = stan_model.optimize(data=data, refresh=1, jacobian=True,
                                    iter=500000, algorithm='lbfgs', seed=seed,
                                    tol_rel_grad=1, output_dir=output_folder)
                    e = False
                except:
                    counter += 1
                    seed = random.randint(1, 2**32-1)
                    shutil.rmtree(output_folder)
            return fit
        
        def sample_one(stan_model, inits, output_folder):
            seed = random.randint(1, 2**32-1)
            counter = 0
            e = True
            while counter<=2000 and e:
                try:
                    fit = stan_model.sampling(data=self.data_sampling, output_dir=output_folder,
                                              iter_warmup=self.iter_warmup, iter_sampling=self.iter_samples,
                                              chains=4, parallel_chains=self.parallel_chains,
                                              max_treedepth=self.max_treedepth,
                                              adapt_delta=self.adapt_delta, seed=seed, inits=inits)
                    e = False
                except:
                    counter += 1
                    seed = random.randint(1, 2**32-1)
                    shutil.rmtree(output_folder)
            return fit
        
        if self.constant_alpha:
            with Pool(4) as pool:
                iters = [[self.MAP_model, f'{self.folder}/{sub_folder}/MAP/a', self.data_MAP[0]], 
                        [self.MAP_model, f'{self.folder}/{sub_folder}/MAP/b', self.data_MAP[1]], 
                        [self.MAP_model, f'{self.folder}/{sub_folder}/MAP/c', self.data_MAP[2]], 
                        [self.MAP_model, f'{self.folder}/{sub_folder}/MAP/d', self.data_MAP[3]]]
                MAP_fit = pool.starmap(find_MAP_one, iters)
                del iters

            inits = [{'F_raw': np.stack([MAP_fit[0].U_raw, MAP_fit[0].V_raw,
                                        MAP_fit[1].U_raw, MAP_fit[1].V_raw,
                                        MAP_fit[2].U_raw, MAP_fit[2].V_raw,
                                        MAP_fit[3].U_raw, MAP_fit[3].V_raw]),
                        'Fv': MAP_fit[i].v_D,
                        'scale': MAP_fit[i].scale} for i in range(4)]
            
            output_folder = [f'{self.folder}/{sub_folder}/Sampling/a',
                                f'{self.folder}/{sub_folder}/Sampling/b',
                                f'{self.folder}/{sub_folder}/Sampling/c',
                                f'{self.folder}/{sub_folder}/Sampling/d']
            
            with Pool(4) as pool:
                pool.starmap(sample_one, [[self.sampling_model, inits[i], output_folder[i]] for i in range(4)])

        else:
            with Pool(5) as pool:
                iters = [[self.MAP_model, f'{self.folder}/{sub_folder}/MAP/a', self.data_MAP[0]], 
                        [self.MAP_model, f'{self.folder}/{sub_folder}/MAP/b', self.data_MAP[1]], 
                        [self.MAP_model, f'{self.folder}/{sub_folder}/MAP/c', self.data_MAP[2]], 
                        [self.MAP_model, f'{self.folder}/{sub_folder}/MAP/d', self.data_MAP[3]],
                        [self.MAP_alpha_model, f'{self.folder}/{sub_folder}/MAP/alpha', self.data_MAP[4]]]
                MAP_fit = pool.starmap(find_MAP_one, iters)
                del iters

            inits = [{'F_raw': np.stack([MAP_fit[0].U_raw, MAP_fit[0].V_raw,
                                        MAP_fit[1].U_raw, MAP_fit[1].V_raw,
                                        MAP_fit[2].U_raw, MAP_fit[2].V_raw,
                                        MAP_fit[3].U_raw, MAP_fit[3].V_raw,
                                        MAP_fit[4].U_raw, MAP_fit[4].V_raw]),
                        'Fv': MAP_fit[i].v_D,
                        'scale': MAP_fit[i].scale} for i in range(5)]

            output_folder = [f'{self.folder}/{sub_folder}/Sampling/a',
                                f'{self.folder}/{sub_folder}/Sampling/b',
                                f'{self.folder}/{sub_folder}/Sampling/c',
                                f'{self.folder}/{sub_folder}/Sampling/d',
                                f'{self.folder}/{sub_folder}/Sampling/alpha']
                
            with Pool(4) as pool:
                pool.starmap(sample_one, [[self.sampling_model, inits[i], output_folder[i]] for i in range(4)])
                
    def __del__(self):
        del self
        



        








        


            
            





        



