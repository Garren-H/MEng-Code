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

class PostProcess:
    def __init__(self, functional_groups, include_clusters, variance_known, inf_type):
        self.functional_groups = functional_groups
        self.include_clusters = include_clusters
        self.variance_known = variance_known
        self.inf_type = inf_type # either 'MAP' or 'Sampling' 
        self.testing_excel = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/TestingData_Final.xlsx' # Path to testing data file

        # set path based on input
        self.init_path = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Hybrid PMF/Subsets'
        self.set_path()
        
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
            csv_files = [f'{self.path}/{i}/{f}' for i in np.sort(os.listdir(self.path)) for f in os.listdir(f'{self.path}/{i}') if f.endswith('.csv')]

            # test csv_files
            MAP = []
            for file in csv_files:
                try:
                    MAP += [cmdstanpy.from_csv(file)]
                except:
                    print(f'Faulty csv file: {file}')
                    print('Skipping...')
            
            A = []

            for i in range(len(MAP)):
                if self.include_clusters:
                    C = np.array(json.load(open(self.data_file, 'r'))['C'])
                    D = json.load(open(self.data_file, 'r'))['D']
                    v_cluster = np.array(json.load(open(self.data_file, 'r'))['v_cluster'])
                    sigma_cluster = (np.sqrt(v_cluster)[np.newaxis,:] * np.ones(D)[:,np.newaxis]) @ C
                    U = sigma_cluster[np.newaxis,:,:] * MAP[i].U_raw + MAP[i].U_raw_means @ C[np.newaxis,:,:]
                    V = sigma_cluster[np.newaxis,:,:] * MAP[i].V_raw + MAP[i].V_raw_means @ C[np.newaxis,:,:]
                
                else:
                    U = MAP[i].U_raw
                    V = MAP[i].V_raw

                v_ARD = np.diag(MAP[i].v_ARD)[np.newaxis, :, :]

                A += [U.transpose(0, 2, 1) @ v_ARD @ V]
            
            A = np.array(A)

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
        # Get the excess enthalpy data
        data = pd.read_excel(self.testing_excel)

        x = data['Composition component 1 [mol/mol]'].to_numpy().astype(float)
        T = data['Temperature [K]'].to_numpy().astype(float)
        y_exp = data['Excess Enthalpy [J/kmol]'].to_numpy().astype(float)
        y_UNIFAC = data['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)
        c1 = data['Component 1'].to_numpy().astype(str)
        c2 = data['Component 2'].to_numpy().astype(str)

        _, _, _, Info_Indices, _, _ = subsets(self.functional_groups).get_subset_df()
        c_all = Info_Indices['Component names']['IUPAC']

        idx1 = np.sum(c1[:, np.newaxis] == c_all[np.newaxis,:], axis=1)
        idx2 = np.sum(c2[:, np.newaxis] == c_all[np.newaxis,:], axis=1)

        idx = (idx1 + idx2) == 2

        data_dict = {'c1': c1[idx],
                     'c2': c2[idx],
                     'x': x[idx],
                     'T': T[idx],
                     'y_exp': y_exp[idx],
                     'y_UNIFAC': y_UNIFAC[idx],}

        _, iidx = np.unique(np.char.add(np.char.add(c1[idx], ' + '), c2[idx]), return_index=True)
        iidx = np.sort(iidx)

        idx1 = np.where(c_all[:, np.newaxis]==c1[idx][iidx][np.newaxis,:])[0]
        idx2 = np.where(c_all[:, np.newaxis]==c2[idx][iidx][np.newaxis,:])[0]

        testing_indices = np.column_stack([idx1, idx2])

        return data_dict, testing_indices
    
    def get_MC(self):
        data_dict, testing_indices = self.get_excess_enthalpy()

        A = self.get_pred_param_matrix()

        scaling = np.array(json.load(open(self.data_file, 'r'))['scaling'])

        _, _, _, Info_Indices, _, _ = subsets(self.functional_groups).get_subset_df()
        c_all = Info_Indices['Component names']['IUPAC']

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])

        y_MC = []
        for i in range(testing_indices.shape[0]):
            mix1 = c_all[testing_indices[i,0]] + ' + ' + c_all[testing_indices[i,1]]
            idx = mix_all == mix1

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
            p12 = A[:, :, Idx_known[i,0], Idx_known[i,1]] * scaling[np.newaxis, :]
            p21 = A[:, :, Idx_known[i,1], Idx_known[i,0]] * scaling[np.newaxis, :]
            xx = x[np.sum(N_points[:i]):np.sum(N_points[:i+1])]
            TT = T[np.sum(N_points[:i]):np.sum(N_points[:i+1])]
            y_rec += [self.excess_enthalpy_predictions(x=xx, 
                                                       T=TT, 
                                                       p12=p12, 
                                                       p21=p21)]

        y_rec = np.concatenate(y_rec, axis=0)

        _, _, _, Info_Indices, _, _ = subsets(self.functional_groups).get_subset_df()
        c_all = Info_Indices['Component names']['IUPAC']

        rec_dict = {'c1': c_all[Idx_known[:,0]],
                    'c2': c_all[Idx_known[:,1]],
                    'x': x,
                    'T': T,
                    'y': y,
                    'y_rec': y_rec}
        
        return rec_dict