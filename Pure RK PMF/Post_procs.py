'''
Function file for the computations of post-processing of Pure results

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
    def __init__(self, functional_groups, include_clusters, variance_known, variance_MC_known):
        self.functional_groups = functional_groups
        self.include_clusters = include_clusters
        self.variance_known = variance_known
        self.variance_MC_known = variance_MC_known
        self.testing_excel = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/TestingData_Final.xlsx' # Path to testing data file

        # set path based on input
        self.path = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Pure RK PMF/Subsets'
        for group in functional_groups:
            if self.path == '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Pure RK PMF/Subsets':
                self.path += '/'+group
            else:
                self.path += '_' + group
        self.data_file = f'{self.path}/data.json' # json data file
        self.path += f'/Include_clusters_{include_clusters}/Variance_known_{variance_known}/Variance_MC_known_{variance_MC_known}/MAP' # path to stan csv files
        # Redlich-Kister polynomial order
        self.order = json.load(open(self.data_file, 'r'))['order']
        # temperature kernel
        self.KT = lambda T1, T2: np.column_stack([np.ones_like(T1), T1**2, 1e-3 * T1 **3]) @ np.column_stack([np.ones_like(T2), T2**2, 1e-3 * T2 **3]).T
        # Composition kernel
        self.Kx = lambda x1, x2: np.column_stack([np.column_stack([x1 ** (self.order+1-i) - x1 for i in range(self.order)]), 
                                                  1e-1 * x1 * np.sqrt(1-x1) * np.exp(x1)]) @ np.column_stack([np.column_stack([x2 ** (self.order+1-i) - x2 for i in range(self.order)]),
                                                                                                             1e-1 * x2 * np.sqrt(1-x2) * np.exp(x2)]).T  
        # Combined kernel
        self.K = lambda x1, x2, T1, T2: self.KT(T1, T2) * self.Kx(x1, x2)

    def get_interps(self, Idx=None):
        if Idx is None: # default to all interps
            Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known'])-1
            Idx_unknown = np.array(json.load(open(self.data_file, 'r'))['Idx_unknown'])-1 
            Idx = np.concatenate([Idx_known, Idx_unknown], axis=0)
            del Idx_known, Idx_unknown
        
        csv_files = [f'{self.path}/{i}/{f}' for i in range(len(os.listdir(self.path))) for f in os.listdir(f'{self.path}/{i}') if f.endswith('.csv')]
        MAP = []
        for file in csv_files:
            MAP += [cmdstanpy.from_csv(file)]

        N_C = json.load(open(self.data_file, 'r'))['N_C']
        N_T = json.load(open(self.data_file, 'r'))['N_T']
        M = int((N_C+1)/2)

        a_rec = []
        for map in MAP:
            #Point estimate reconstructed values
            a_rec += [np.concatenate([np.concatenate([np.concatenate([np.stack([(map.U_raw[t,m,:,:].T @ np.diag(map.v_ARD) @ map.V_raw[t,m,:,:])[Idx[:,0], Idx[:,1]] for m in range(M-1)], axis=0 ), 
                            ((map.U_raw[t,-1,:,:].T @ np.diag(map.v_ARD) @ map.U_raw[t,-1,:,:])[Idx[:,0], Idx[:,1]][np.newaxis,:])], axis=0), 
                            np.stack([(map.U_raw[t,m,:,:].T @ np.diag(map.v_ARD) @ map.V_raw[t,m,:,:])[Idx[:,1], Idx[:,0]] for m in range(M-1)], axis=0 )[::-1,:]], axis=0) for t in range(N_T)], axis=0)]

        return np.stack(a_rec) # returns tensor of size len(MAP) x (N_T*N_C) x (N*(N-1)/2)

    def compute_testing_error(self):
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])
        N_C = json.load(open(self.data_file, 'r'))['N_C']
        N_T = json.load(open(self.data_file, 'r'))['N_T']
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        if self.variance_MC_known:
            v_MC = json.load(open(self.data_file, 'r'))['v_MC']
        else:
            v_MC = []
            csv_files = [f'{self.path}/{i}/{f}' for i in range(len(os.listdir(self.path))) for f in os.listdir(f'{self.path}/{i}') if f.endswith('.csv')]
            for file in csv_files:
                MAP = cmdstanpy.from_csv(file)
                v_MC += [MAP.v_MC]
            v_MC = np.array(v_MC)

        x2 = np.concatenate([x2_int for _ in range(N_T)])
        T2 = np.concatenate([T2_int[t]*np.ones_like(x2_int) for t in range(N_T)])

        # Get the excess enthalpy data
        data = pd.read_excel(self.testing_excel)

        x = data['Composition component 1 [mol/mol]'].to_numpy().astype(float)
        T = data['Temperature [K]'].to_numpy().astype(float)
        y_exp = data['Excess Enthalpy [J/mol]'].to_numpy().astype(float)
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

        a_rec_testing = self.get_interps(Idx=testing_indices)

        y_MC = []

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])

        for i in range(testing_indices.shape[0]):
            yy_MC = []
            mix1 = c_all[testing_indices[i,0]] + ' + ' + c_all[testing_indices[i,1]]
            idx = mix_all == mix1
            K_test_interp = self.K(data_dict['x'][idx], x2, data_dict['T'][idx], T2)
            K_interp = self.K(x2, x2, T2, T2) + (jitter) * np.eye(len(x2))
            
            for j in range(a_rec_testing.shape[0]):
                try:
                    cov_interp_inv = np.linalg.inv(K_interp+v_MC[j]*np.eye(K_interp.shape[0])).T
                except:
                    cov_interp_inv = np.linalg.inv(K_interp+v_MC*np.eye(K_interp.shape[0])).T
                yy_MC += [K_test_interp @ cov_interp_inv @ a_rec_testing[j,:,i]]
            
            yy_MC = np.stack(yy_MC, axis=0)
            y_MC += [yy_MC]
        
        data_dict['y_MC'] = np.concatenate(y_MC, axis=1).T

        return data_dict
    
    def compute_reconstuction_error(self):
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])
        N_C = json.load(open(self.data_file, 'r'))['N_C']
        N_T = json.load(open(self.data_file, 'r'))['N_T']
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known'])-1
        N_points = np.array(json.load(open(self.data_file, 'r'))['N_points'])
        if self.variance_MC_known:
            v_MC = json.load(open(self.data_file, 'r'))['v_MC']
        else:
            v_MC = []
            csv_files = [f'{self.path}/{i}/{f}' for i in range(len(os.listdir(self.path))) for f in os.listdir(f'{self.path}/{i}') if f.endswith('.csv')]
            for file in csv_files:
                MAP = cmdstanpy.from_csv(file)
                v_MC += [MAP.v_MC]
            v_MC = np.array(v_MC)

        x2 = np.concatenate([x2_int for _ in range(N_T)])
        T2 = np.concatenate([T2_int[t]*np.ones_like(x2_int) for t in range(N_T)])

        data_dict = {'x': np.array(json.load(open(self.data_file, 'r'))['x1']),
                     'T': np.array(json.load(open(self.data_file, 'r'))['T1']),
                     'y': np.array(json.load(open(self.data_file, 'r'))['y1'])}

        a_rec_testing = self.get_interps(Idx=Idx_known)

        y_MC = []

        for i in range(Idx_known.shape[0]):
            yy_MC = []
            K_test_interp = self.K(data_dict['x'][np.sum(N_points[:i]):np.sum(N_points[:i+1])], x2, data_dict['T'][np.sum(N_points[:i]):np.sum(N_points[:i+1])], T2)
            K_interp = self.K(x2, x2, T2, T2) + (jitter) * np.eye(len(x2))
            
            for j in range(a_rec_testing.shape[0]):
                try:
                    L_inv_T = np.linalg.inv(np.linalg.cholesky(K_interp+v_MC[j]*np.eye(K_interp.shape[0]))).T
                except:
                    L_inv_T = np.linalg.inv(np.linalg.cholesky(K_interp+v_MC*np.eye(K_interp.shape[0]))).T
                yy_MC += [K_test_interp @ (L_inv_T@L_inv_T.T) @a_rec_testing[j,:,i]]
            
            yy_MC = np.stack(yy_MC, axis=0)
            y_MC += [yy_MC]
        
        data_dict['y_MC'] = np.concatenate(y_MC, axis=1).T

        return data_dict
    
    def compute_error_metrics(self, data_dict=None):
        if data_dict == None:
            # Default to testing data
            data_dict = self.compute_testing_error()

        y_exp = data_dict['y_exp']
        y_UNIFAC = data_dict['y_UNIFAC']
        y_MC = data_dict['y_MC']

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        err_dict = {('Component 1', ''): [],
                    ('Component 2', ''): [],
                    ('UNIFAC', 'MAE'): [],
                    ('UNIFAC', 'RMSE'): [],
                    ('MC', 'MAE'): [],
                    ('MC', 'RMSE'): []}

        for j in range(len(unique_mix)):
            idx = mix_all == unique_mix[j]
            err_dict['Component 1', ''] += [data_dict['c1'][idx][0]]
            err_dict['Component 2', ''] += [data_dict['c2'][idx][0]]
            err_dict['UNIFAC', 'MAE'] += [np.mean(np.abs(y_exp[idx] - y_UNIFAC[idx]))]
            err_dict['MC', 'MAE'] += [np.mean(np.abs(y_exp[idx] - y_MC[idx]))]
            err_dict['UNIFAC', 'RMSE'] += [np.sqrt(np.mean((y_exp[idx] - y_UNIFAC[idx])**2))]
            err_dict['MC', 'RMSE'] += [np.sqrt(np.mean((y_exp[idx] - y_MC[idx])**2))]

        err_dict['Component 1', ''] += ['Overall']
        err_dict['Component 2', ''] += ['']
        err_dict['UNIFAC', 'MAE'] += [np.mean(np.abs(y_exp - y_UNIFAC))]
        err_dict['MC', 'MAE'] += [np.mean(np.abs(y_exp - y_MC))]
        err_dict['UNIFAC', 'RMSE'] += [np.sqrt(np.mean((y_exp - y_UNIFAC)**2))]
        err_dict['MC', 'RMSE'] += [np.sqrt(np.mean((y_exp - y_MC)**2))]

        return err_dict