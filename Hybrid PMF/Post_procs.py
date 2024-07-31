'''
Function file for the ocompiutation of post-processing of Hybrid results

The functions essetially just computes the error between the testing excess enthalpy data
and the experimental data.


'''

import numpy as np
import pandas as pd
import cmdstanpy
import os
import json

from All_code import subsets

import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class PostProcess:
    def __init__(self, functional_groups, include_clusters, variance_known, inf_type):
        self.functional_groups = functional_groups
        self.include_clusters = include_clusters
        self.variance_known = variance_known
        self.inf_type = inf_type # either 'MAP' or 'Sampling' 
        self.testing_excel = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/TestingData_Final.xlsx' # Path to testing data file
        self.training_excel = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Sorted Data.xlsx' # Path to training data file

        # set path based on input
        self.init_path = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Hybrid PMF/Subsets'
        self.set_path()

        self.c_all = subsets(self.functional_groups).get_subset_df()[3]['Component names']['IUPAC'].astype(str)
        
        Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known']) - 1
        self.Sparse = np.zeros((np.max(Idx_known)+1,np.max(Idx_known)+1)).astype(int)
        for i in range(len(Idx_known)):
            self.Sparse[Idx_known[i,0], Idx_known[i,1]] = 1
            self.Sparse[Idx_known[i,1], Idx_known[i,0]] = 1
        
    def set_path(self): # sets the path initially and if some inputs are changed
        self.path = self.init_path
        for group in self.functional_groups:
            if self.path == self.init_path:
                self.path += f'/{group}'
            else:
                self.path += f'_{group}'
        self.data_file = f'{self.path}/data.json'
        self.path += f'/Include_clusters_{self.include_clusters}/Variance_known_{self.variance_known}/{self.inf_type}'
    
    def reset_init_path(self, path): # Overwrite init_path if results stored in a different folder
        self.init_path = path
        self.set_path(self)   

    def reset_include_clusters(self, include_clusters): # Change the include_clusters parameter without having to redefine the class
        self.include_clusters = include_clusters
        self.set_path(self)

    def reset_variance_known(self, variance_known): # Change the variance_known parameter without having to redefine the class
        self.variance_known = variance_known
        self.set_path(self)

    def reset_inf_type(self, inf_type): # Change the inf_type parameter without having to redefine the class
        self.inf_type = inf_type
        self.set_path(self)

    def excess_enthalpy_predictions(self, x, T, p12, p21, a=0.3):
        # Function to compute the excess enthalpy predictions
        if p12.ndim > 1:
            x = x[:, np.newaxis]
            if not np.isscalar(T):
                T = T[:, np.newaxis]
            
            t12 = p12[:,0][np.newaxis,:] + p12[:,1][np.newaxis,:] * T + p12[:,2][np.newaxis,:] / T + p12[:,3][np.newaxis,:] * np.log(T)
            t21 = p21[:,0][np.newaxis,:] + p21[:,1][np.newaxis,:] * T + p21[:,2][np.newaxis,:] / T + p21[:,3][np.newaxis,:] * np.log(T)
            dt12_dT = p12[:,1][np.newaxis,:] - p12[:,2][np.newaxis,:] / T**2 + p12[:,3][np.newaxis,:] / T
            dt21_dT = p21[:,1][np.newaxis,:] - p21[:,2][np.newaxis,:] / T**2 + p21[:,3][np.newaxis,:] / T

        else:
            t12 = p12[0] + p12[1] * T + p12[2] / T + p12[3] * np.log(T)
            t21 = p21[0] + p21[1] * T + p21[2] / T + p21[3] * np.log(T)
            dt12_dT = p12[1] - p12[2] / T**2 + p12[3] / T
            dt21_dT = p21[1] - p21[2] / T**2 + p21[3] / T

        G12 = np.exp(-a*t12)
        G21 = np.exp(-a*t21)

        term1 = ( ( (1-x) * G12 * (1 - a*t12) + x * G12**2 ) / ((1-x) + x * G12)**2 ) * dt12_dT
        term2 = ( ( x * G21 * (1 - a*t21) + (1-x) * G21**2 ) / (x + (1-x) * G21)**2 ) * dt21_dT
        
        return -8.314 * T**2 * x * (1-x) * ( term1 + term2 )

    def get_pred_param_matrix(self):
        N = json.load(open(self.data_file, 'r'))['N']

        # Load the stan csv files
        if self.inf_type == 'MAP':
            # get csv_files
            csv_files = [
                f'{self.path}/{i}/{f}' 
                for i in np.sort(os.listdir(self.path)) 
                if i.isdigit()  # Check if the directory name is an integer
                for f in os.listdir(f'{self.path}/{i}') 
                if f.endswith('.csv')
            ]

            # test csv_files
            MAP = []
            for file in csv_files:
                try:
                    MAP += [cmdstanpy.from_csv(file)]
                except:
                    print(f'Faulty csv file: {file}')
                    print('Skipping...')
            
            # Extract max lp
            max_lp = np.argmax([map.optimized_params_dict['lp__'] for map in MAP])
            MAP = MAP[max_lp]

            if self.include_clusters:
                C = np.array(json.load(open(self.data_file, 'r'))['C'])
                D = json.load(open(self.data_file, 'r'))['D']
                v_cluster = np.array(json.load(open(self.data_file, 'r'))['v_cluster'])
                sigma_cluster = (np.sqrt(v_cluster)[np.newaxis,:] * np.ones(D)[:,np.newaxis]) @ C
                U = sigma_cluster[np.newaxis,:,:] * MAP.U_raw + MAP.U_raw_means @ C[np.newaxis,:,:]
                V = sigma_cluster[np.newaxis,:,:] * MAP.V_raw + MAP.V_raw_means @ C[np.newaxis,:,:]
            
            else:
                U = MAP.U_raw
                V = MAP.V_raw

            v_ARD = np.diag(MAP.v_ARD)[np.newaxis, :, :]

            A = U.transpose(0, 2, 1) @ v_ARD @ V

        elif self.inf_type == 'Sampling':
            # get csv_files
            csv_files = [f'{self.path}/{f}' for f in os.listdir(f'{self.path}') if f.endswith('.csv')]

            # test csv_files
            idx_remove = []
            counter = 0
            for file in csv_files:
                try:
                    fit = cmdstanpy.from_csv(file)
                except:
                    print(f'Faulty csv file: {file}')
                    print('Skipping...')
                    idx_remove += [counter]
                counter += 1
            
            csv_files = [csv_files[i] for i in range(len(csv_files)) if i not in idx_remove]

            fit = cmdstanpy.from_csv(csv_files)

            D = json.load(open(self.data_file, 'r'))['D']
            if self.include_clusters:
                C = np.array(json.load(open(self.data_file, 'r'))['C'])
                v_cluster = np.array(json.load(open(self.data_file, 'r'))['v_cluster'])
                sigma_cluster = (np.sqrt(v_cluster)[np.newaxis,:] * np.ones(D)[:,np.newaxis]) @ C
                U = sigma_cluster[np.newaxis, np.newaxis,:,:] * fit.U_raw + fit.U_raw_means @ C[np.newaxis, np.newaxis,:,:]
                V = sigma_cluster[np.newaxis, np.newaxis,:,:] * fit.V_raw + fit.V_raw_means @ C[np.newaxis, np.newaxis,:,:]

            else:
                U = fit.U_raw
                V = fit.V_raw
            
            v_ARD = (fit.v_ARD[:, np.newaxis, :] * np.eye(D)[np.newaxis, :, :])[:, np.newaxis, :, :]

            A = U.transpose(0, 1, 3, 2) @ v_ARD @ V

        else:
            print('inf_type not recognized')
            print('Specify either "MAP" or "Sampling"')
            return None

        return A        

    def get_excess_enthalpy(self):
        N = json.load(open(self.data_file, 'r'))['N']
        # Get the excess enthalpy data
        data = pd.read_excel(self.testing_excel)

        x = data['Composition component 1 [mol/mol]'].to_numpy().astype(float)
        T = data['Temperature [K]'].to_numpy().astype(float)
        y_exp = data['Excess Enthalpy [J/mol]'].to_numpy().astype(float)
        y_UNIFAC = data['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)
        c1 = data['Component 1'].to_numpy().astype(str)
        c2 = data['Component 2'].to_numpy().astype(str)

        idx1 = np.sum(c1[:, np.newaxis] == self.c_all[np.newaxis,:], axis=1)
        idx2 = np.sum(c2[:, np.newaxis] == self.c_all[np.newaxis,:], axis=1)

        idx = (idx1 + idx2) == 2

        data_dict = {'c1': c1[idx],
                     'c2': c2[idx],
                     'x': x[idx],
                     'T': T[idx],
                     'y_exp': y_exp[idx],
                     'y_UNIFAC': y_UNIFAC[idx],}

        _, iidx = np.unique(np.char.add(np.char.add(c1[idx], ' + '), c2[idx]), return_index=True)
        iidx = np.sort(iidx)

        idx1 = np.sum((np.arange(N)[:, np.newaxis] * (self.c_all[:, np.newaxis]==c1[idx][iidx][np.newaxis,:])), axis=0)
        idx2 = np.sum((np.arange(N)[:, np.newaxis] * (self.c_all[:, np.newaxis]==c2[idx][iidx][np.newaxis,:])), axis=0)

        testing_indices = np.column_stack([idx1, idx2])

        return data_dict, testing_indices
    
    def get_MC(self):
        data_dict, testing_indices = self.get_excess_enthalpy()

        A = self.get_pred_param_matrix()

        scaling = np.array(json.load(open(self.data_file, 'r'))['scaling'])

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])

        y_MC = []
        for i in range(testing_indices.shape[0]):
            mix1 = self.c_all[testing_indices[i,0]] + ' + ' + self.c_all[testing_indices[i,1]]
            idx = mix_all == mix1

            if self.inf_type == 'MAP':
                p12 = A[:, testing_indices[i,0], testing_indices[i,1]] * scaling
                p21 = A[:, testing_indices[i,1], testing_indices[i,0]] * scaling

            elif self.inf_type == 'Sampling':
                p12 = A[:, :, testing_indices[i,0], testing_indices[i,1]] * scaling[np.newaxis, :]
                p21 = A[:, :, testing_indices[i,1], testing_indices[i,0]] * scaling[np.newaxis, :]

            y_MC += [self.excess_enthalpy_predictions(x=data_dict['x'][idx], 
                                                      T=data_dict['T'][idx], 
                                                      p12=p12, 
                                                      p21=p21)]

        y_MC = np.concatenate(y_MC, axis=0)

        data_dict['y_MC'] = y_MC

        return data_dict

    def get_reconstructed_errors(self):

        A = self.get_pred_param_matrix()

        with open(self.data_file, 'r') as f:
            data = json.load(f)
        x = np.array(data['x'])
        T = np.array(data['T'])
        y = np.array(data['y'])
        Idx_known = np.array(data['Idx_known'])-1
        N_points = np.array(data['N_points'])
        N_known = np.array(data['N_known'])
        scaling = np.array(data['scaling'])

        y_rec = []

        for i in range(N_known):
            if self.inf_type == 'MAP':
                p12 = A[:, Idx_known[i,0], Idx_known[i,1]] * scaling
                p21 = A[:, Idx_known[i,1], Idx_known[i,0]] * scaling
            
            elif self.inf_type == 'Sampling':
                p12 = A[:, :, Idx_known[i,0], Idx_known[i,1]] * scaling[np.newaxis, :]
                p21 = A[:, :, Idx_known[i,1], Idx_known[i,0]] * scaling[np.newaxis, :]

            xx = x[np.sum(N_points[:i]):np.sum(N_points[:i+1])]
            TT = T[np.sum(N_points[:i]):np.sum(N_points[:i+1])]
            y_rec += [self.excess_enthalpy_predictions(x=xx, 
                                                       T=TT, 
                                                       p12=p12, 
                                                       p21=p21)]

        y_rec = np.concatenate(y_rec, axis=0)

        y_UNI = subsets(self.functional_groups).get_subset_df()[0]['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)
        subset_Indices_T = subsets(self.functional_groups).get_subset_df()[2]

        y_UNIFAC = np.concatenate([y_UNI[subset_Indices_T[j,0]:subset_Indices_T[j,1]+1] for j in range(subset_Indices_T.shape[0])])

        rec_dict = {'c1': np.concatenate([[self.c_all[Idx_known[i,0]].astype(str)]*N_points[i] for i in range(len(Idx_known))]),
                    'c2': np.concatenate([[self.c_all[Idx_known[i,1]].astype(str)]*N_points[i] for i in range(len(Idx_known))]),
                    'x': x,
                    'T': T,
                    'y_exp': y,
                    'y_UNIFAC': y_UNIFAC,
                    'y_MC': y_rec}
        
        return rec_dict
    
    def get_num_known_mix(self, c):
        idx = np.arange(len(self.c_all))[self.c_all == c]

        return np.sum(self.Sparse[:,idx])
    
    def get_num_common_mix(self, c1, c2):
        idx1 = np.arange(len(self.c_all))[self.c_all == c1]
        idx2 = np.arange(len(self.c_all))[self.c_all == c2]

        return np.sum(self.Sparse[:,idx1] * self.Sparse[:,idx2])
    
    def compute_error_metrics(self, data_dict=None, is_testing=None):
        if data_dict == None:
            # Default to testing data
            data_dict = self.get_MC()
            is_testing = True

        if type(is_testing) != bool:
            raise ValueError('is_testing must be a booleans')

        y_exp = data_dict['y_exp']
        y_UNIFAC = data_dict['y_UNIFAC']
        if self.inf_type == 'Sampling':
            y_MC = np.mean(data_dict['y_MC'], axis=1)
        elif self.inf_type == 'MAP':
            y_MC = data_dict['y_MC']

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        err_dict = {('IUPAC', 'Component 1'): [],
                    ('IUPAC', 'Component 2'): [],
                    ('Number of known mixtures', 'Component 1'): [],
                    ('Number of known mixtures', 'Component 2'): [],
                    ('Number of common mixture', ''): [],
                    ('Number of datapoints', ''): [],
                    ('UNIFAC', 'MAE'): [],
                    ('UNIFAC', 'RMSE'): [],}
        if is_testing:
            err_dict['UNIFAC', 'MARE'] = []
        
        err_dict['MC', 'MAE'] = []
        err_dict['MC', 'RMSE'] = []
        
        if is_testing:
            err_dict['MC', 'MARE'] = []

        for j in range(len(unique_mix)):
            idx = mix_all == unique_mix[j]
            err_dict['IUPAC', 'Component 1'] += [data_dict['c1'][idx][0]]
            err_dict['IUPAC', 'Component 2'] += [data_dict['c2'][idx][0]]
            err_dict['Number of known mixtures', 'Component 1'] += [self.get_num_known_mix(data_dict['c1'][idx][0])]
            err_dict['Number of known mixtures', 'Component 2'] += [self.get_num_known_mix(data_dict['c2'][idx][0])]
            err_dict['Number of common mixture', ''] += [self.get_num_common_mix(data_dict['c1'][idx][0], data_dict['c2'][idx][0])]
            err_dict['UNIFAC', 'MAE'] += [np.mean(np.abs(y_exp[idx] - y_UNIFAC[idx]))]
            err_dict['MC', 'MAE'] += [np.mean(np.abs(y_exp[idx] - y_MC[idx]))]
            err_dict['UNIFAC', 'RMSE'] += [np.sqrt(np.mean((y_exp[idx] - y_UNIFAC[idx])**2))]
            err_dict['MC', 'RMSE'] += [np.sqrt(np.mean((y_exp[idx] - y_MC[idx])**2))]
            err_dict['Number of datapoints', ''] += [np.sum(idx)]

            if is_testing:
                err_dict['UNIFAC', 'MARE'] += [np.mean(np.abs((y_exp[idx] - y_UNIFAC[idx]) / y_exp[idx]))]
                err_dict['MC', 'MARE'] += [np.mean(np.abs((y_exp[idx] - y_MC[idx]) / y_exp[idx]))]

        err_dict['IUPAC', 'Component 1'] += ['Overall']
        err_dict['IUPAC', 'Component 2'] += ['']
        err_dict['Number of known mixtures', 'Component 1'] += ['']
        err_dict['Number of known mixtures', 'Component 2'] += ['']
        err_dict['Number of common mixture', ''] += ['']
        err_dict['UNIFAC', 'MAE'] += [np.mean(np.abs(y_exp - y_UNIFAC))]
        err_dict['MC', 'MAE'] += [np.mean(np.abs(y_exp - y_MC))]
        err_dict['UNIFAC', 'RMSE'] += [np.sqrt(np.mean((y_exp - y_UNIFAC)**2))]
        err_dict['MC', 'RMSE'] += [np.sqrt(np.mean((y_exp - y_MC)**2))]
        err_dict['Number of datapoints', ''] += [len(y_exp)]

        if is_testing:
            err_dict['UNIFAC', 'MARE'] += [np.mean(np.abs((y_exp - y_UNIFAC) / y_exp))]
            err_dict['MC', 'MARE'] += [np.mean(np.abs((y_exp - y_MC) / y_exp))]

        return pd.DataFrame(err_dict)

    def plot_2D_pred(self, data_dict=None, is_testing=None):
        if data_dict == None:
            # Default to testing data
            data_dict = self.get_MC()
            is_testing = True
        if type(is_testing) != bool:
            raise ValueError('is_testing must be a boolean')
        
        excel_UNI_plot = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/UNIFAC_Plots.xlsx'
        if is_testing:
            path = f'{self.path}/2D_plots/Testing'
            excel_UNI_sheet = 'Testing_Plots'
            _, Idx_known = self.get_excess_enthalpy()
        else:
            path = f'{self.path}/2D_plots/Training'
            excel_UNI_sheet = 'Training_Plots'
            Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known']) - 1

        try:
            os.makedirs(path)
        except:
            print(f'Directory {path} already exists')
            
        A = self.get_pred_param_matrix()
        
        all_mix = np.char.add(np.char.add(data_dict['c1'].astype(str), ' + '), data_dict['c2'].astype(str))
        unique_mix, idx = np.unique(all_mix, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        df_UNIFAC = pd.read_excel(excel_UNI_plot, sheet_name=excel_UNI_sheet)

        exp_mix = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])
        UNIFAC_mix = np.char.add(np.char.add(df_UNIFAC['Component 1'].to_numpy().astype(str), ' + '), df_UNIFAC['Component 2'].to_numpy().astype(str))

        for j in range(len(unique_mix)):
            y_idx = exp_mix == unique_mix[j]
            UNIFAC_idx = UNIFAC_mix == unique_mix[j]
            yy = data_dict['y_exp'][y_idx]
            yy_UNIFAC = df_UNIFAC['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[UNIFAC_idx]
            x_y = data_dict['x'][y_idx]
            T_y = data_dict['T'][y_idx]
            c1 = data_dict['c1'][y_idx][0]
            c2 = data_dict['c2'][y_idx][0]

            x_UNIFAC = df_UNIFAC['Composition component 1 [mol/mol]'].to_numpy().astype(float)[UNIFAC_idx]
            T_UNIFAC = df_UNIFAC['Temperature [K]'].to_numpy().astype(float)[UNIFAC_idx]

            if self.inf_type == 'MAP':
                p12 = A[:, Idx_known[j,0], Idx_known[j,1]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])
                p21 = A[:, Idx_known[j,1], Idx_known[j,0]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])
                yy_MC_mean = self.excess_enthalpy_predictions(x=x_UNIFAC, T=T_UNIFAC, p12=p12, p21=p21)

            elif self.inf_type == 'Sampling':
                p12 = A[:, :, Idx_known[j,0], Idx_known[j,1]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])[np.newaxis, :]
                p21 = A[:, :, Idx_known[j,1], Idx_known[j,0]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])[np.newaxis, :]
                yy_MC = self.excess_enthalpy_predictions(x=x_UNIFAC, T=T_UNIFAC, p12=p12, p21=p21)
                yy_MC_mean = np.mean(yy_MC, axis=1)
                yy_MC_0025 = np.quantile(yy_MC, 0.025, axis=1)
                yy_MC_0975 = np.quantile(yy_MC, 0.975, axis=1)

            T_uniq = np.unique(T_UNIFAC)
            for i in range(len(T_uniq)):
                if not os.path.exists(f'{path}/{j}_{i}.png'):
                    print(f'{j+1} out of {len(unique_mix)} mixtures')
                    print(f'{i+1} out of {len(T_uniq)} temperatures')
                    TT = T_uniq[i]
                    T_y_idx = np.abs(T_y - TT) <= 0.5
                    T_UNIFAC_idx = T_UNIFAC == TT

                    fig, ax = plt.subplots()
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_UNIFAC[T_UNIFAC_idx], '-g', label='UNIFAC')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx], '-b', label='Mean MC')
                    if self.inf_type == 'Sampling':
                        ax.fill_between(x_UNIFAC[T_UNIFAC_idx], yy_MC_0025[T_UNIFAC_idx], yy_MC_0975[T_UNIFAC_idx], color='b', alpha=0.5, label='95% CI MC')
                    ax.plot(x_y[T_y_idx], yy[T_y_idx], '.k', label='Experimental Data')
                    ax.set_xlabel('Composition of Compound 1 [mol/mol]')
                    ax.set_ylabel('Excess Enthalpy [J/mol]')
                    ax.set_title(f'(1) {c1} + (2) {c2} at {T_uniq[i]:.2f} K')
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout()

                    fig.savefig(f'{path}/{j}_{i}.png', dpi=300)
                    plt.close(fig)
                    clear_output(wait=False)

    def plot_3D_pred(self, data_dict=None, is_testing=None):
        if data_dict == None:
            # Default to testing data
            data_dict = self.get_MC()
            is_testing = True
        if type(is_testing) != bool:
            raise ValueError('is_testing must be a boolean')
        
        excel_UNI_plot = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/UNIFAC_Plots.xlsx'
        if is_testing:
            path = f'{self.path}/3D_plots/Testing'
            excel_UNI_sheet = 'Testing_Plots'
            _, Idx_known = self.get_excess_enthalpy()
        else:
            path = f'{self.path}/3D_plots/Training'
            excel_UNI_sheet = 'Training_Plots'
            Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known']) - 1

        try:
            os.makedirs(path)
        except:
            print(f'Directory {path} already exists')
            
        A = self.get_pred_param_matrix()
        
        all_mix = np.char.add(np.char.add(data_dict['c1'].astype(str), ' + '), data_dict['c2'].astype(str))
        unique_mix, idx = np.unique(all_mix, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        df_UNIFAC = pd.read_excel(excel_UNI_plot, sheet_name=excel_UNI_sheet)

        exp_mix = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])
        UNIFAC_mix = np.char.add(np.char.add(df_UNIFAC['Component 1'].to_numpy().astype(str), ' + '), df_UNIFAC['Component 2'].to_numpy().astype(str))

        for j in range(len(unique_mix)):
            if not os.path.exists(f'{path}/{j}.png'):
                print(f'{j+1} out of {len(unique_mix)} mixtures')
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                y_idx = exp_mix == unique_mix[j]
                UNIFAC_idx = UNIFAC_mix == unique_mix[j]
                yy = data_dict['y_exp'][y_idx]
                yy_UNIFAC = df_UNIFAC['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[UNIFAC_idx]
                x_y = data_dict['x'][y_idx]
                T_y = data_dict['T'][y_idx]
                c1 = data_dict['c1'][y_idx][0]
                c2 = data_dict['c2'][y_idx][0]

                x_UNIFAC = df_UNIFAC['Composition component 1 [mol/mol]'].to_numpy().astype(float)[UNIFAC_idx]
                T_UNIFAC = df_UNIFAC['Temperature [K]'].to_numpy().astype(float)[UNIFAC_idx]

                if self.inf_type == 'MAP':
                    p12 = A[:, Idx_known[j,0], Idx_known[j,1]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])
                    p21 = A[:, Idx_known[j,1], Idx_known[j,0]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])
                    yy_MC_mean = self.excess_enthalpy_predictions(x=x_UNIFAC, T=T_UNIFAC, p12=p12, p21=p21)

                elif self.inf_type == 'Sampling':
                    p12 = A[:, :, Idx_known[j,0], Idx_known[j,1]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])[np.newaxis, :]
                    p21 = A[:, :, Idx_known[j,1], Idx_known[j,0]] * np.array(json.load(open(self.data_file, 'r'))['scaling'])[np.newaxis, :]
                    yy_MC = self.excess_enthalpy_predictions(x=x_UNIFAC, T=T_UNIFAC, p12=p12, p21=p21)
                    yy_MC_mean = np.mean(yy_MC, axis=1)
                    yy_MC_0025 = np.quantile(yy_MC, 0.025, axis=1)
                    yy_MC_0975 = np.quantile(yy_MC, 0.975, axis=1)

                T_uniq = np.unique(T_UNIFAC)
                for i in range(len(T_uniq)):
                    # Plot median prediction
                    TT = T_uniq[i]
                    T_y_idx = np.abs(T_y - TT) <= 0.5
                    T_UNIFAC_idx = T_UNIFAC == TT
                    if j == 0:
                        ax.plot(x_UNIFAC[T_UNIFAC_idx], T_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx], c='b', label='Mean MC')
                    else:
                        ax.plot(x_UNIFAC[T_UNIFAC_idx], T_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx], c='b')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], T_UNIFAC[T_UNIFAC_idx], yy_UNIFAC[T_UNIFAC_idx], c='g', label='UNIFAC')
                    # Create polygons for CI bounds
                    if self.inf_type == 'Sampling':
                        verts = [list(zip(x_UNIFAC[T_UNIFAC_idx], T_UNIFAC[T_UNIFAC_idx], yy_MC_0025[T_UNIFAC_idx])) + list(zip(x_UNIFAC[T_UNIFAC_idx], T_UNIFAC[T_UNIFAC_idx], yy_MC_0975[T_UNIFAC_idx]))[::-1]]
                        poly = Poly3DCollection(verts, facecolors='b', alpha=0.5)
                        ax.add_collection3d(poly)

                # Scatter plot for experimental data
                ax.scatter(x_y, T_y, yy, c='k', marker='.', s=100, label='Experimental Data')

                # Custom legend
                custom_lines = [
                    Line2D([0], [0], color='k', marker='.', linestyle='None', markersize=10),  # Experimental Data
                    Line2D([0], [0], color='g', lw=4),  # UNIFAC
                    Line2D([0], [0], color='b', lw=4) # Mean MC
                ]

                if self.inf_type == 'Sampling':
                    custom_lines += [Line2D([0], [0], color='b', lw=4, alpha=0.5)],  # 95% CI Bounds
                    ax.legend(custom_lines, ['Experimental Data', 'UNIFAC', 'Mean MC', '95% CI MC'], loc='upper left', bbox_to_anchor=(1.03, 1))
                elif self.inf_type == 'MAP':
                    ax.legend(custom_lines, ['Experimental Data', 'UNIFAC', 'Mean MC'], loc='upper left', bbox_to_anchor=(1.03, 1))
                    
                ax.set_xlabel('Composition of component 1 [mol//mol]', fontsize=14)
                ax.set_ylabel('Temperature [K]', fontsize=14)
                ax.set_zlabel('Excess Enthalpy [J/mol]', fontsize=14, labelpad=10)
                ax.set_title(f'(1) {c1} + (2) {c2}', fontsize=20)
                plt.tight_layout()  # Adjust layout to make room for the legend

                plt.savefig(f'{path}/{j}.png', dpi=300)

                plt.close(fig)
                clear_output(wait=False)


