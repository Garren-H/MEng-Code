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
    def __init__(self, functional_groups, include_clusters: bool, add_zeros: bool, refT: bool, ARD: bool):
        self.functional_groups = functional_groups
        self.include_clusters = include_clusters
        self.add_zeros = add_zeros
        self.refT = refT
        self.ARD = ARD
        self.testing_excel = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/TestingData_Final.xlsx' # Path to testing data file

        # set path based on input
        self.path = f'Subsets/'
        self.path += self.functional_groups[0]
        for functional_group in self.functional_groups[1:]:
            self.path += f'_{functional_group}'
        self.data_file = f'{self.path}/data.json'
        self.path += f'/Include_clusters_{self.include_clusters}/Add_zeros_{self.add_zeros}/RefT_{self.refT}/ARD_{self.ARD}/MAP'
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

        # All components
        _, _, _, Info_Indices, _, _ = subsets(self.functional_groups).get_subset_df()
        self.c_all = Info_Indices['Component names']['IUPAC']

    def get_interps(self, Idx=None):
        if Idx is None: # default to all interps
            Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known'])-1
            Idx_unknown = np.array(json.load(open(self.data_file, 'r'))['Idx_unknown'])-1 
            Idx = np.concatenate([Idx_known, Idx_unknown], axis=0)
            del Idx_known, Idx_unknown
        
        path_to_files = [f'{self.path}/{i}' for i in os.listdir(self.path) if i.isdigit()]
        MAP = []
        for path in path_to_files:
            try: # load csv files
                csv_file = [f'{path}/{f}' for f in os.listdir(path) if f.endswith('.csv')][0]
                MAP_rec = cmdstanpy.from_csv(csv_file)
                keys = list(MAP_rec[-1].stan_variables().keys())
                for key in keys:
                    try:
                        MAP[-1][key] = MAP_rec[-1].stan_variables()[key]
                    except:
                        pass
                del MAP_rec
            except: # load json files if csv file is faulty
                inits_file = f'{path}/inits.json'
                MAP += [json.load(open(inits_file, 'r'))]
                # convert lists to numpy arrays
                keys = list(MAP[-1].keys())
                for key in keys:
                    try:
                        MAP[-1][key] = np.array(MAP[-1][key])
                    except:
                        pass
        
        N = json.load(open(self.data_file, 'r'))['N']
        N_C = json.load(open(self.data_file, 'r'))['N_C']
        N_T = json.load(open(self.data_file, 'r'))['N_T']
        M = int(N_C/2)

        if self.include_clusters:   
            C = np.array(json.load(open(self.data_file, 'r'))['C'])
            v_cluster = np.array(json.load(open(self.data_file, 'r'))['v_cluster'])
        if self.refT:
            sigma_refT = np.array(json.load(open(self.data_file, 'r'))['sigma_refT'])

        a_rec = []
        for map in MAP:
            D = map['U_raw'].shape[2]
            #Point estimate reconstructed values
            if self.include_clusters and not self.ref_temp:
                sigma_cluster = (np.sqrt(v_cluster)[np.newaxis,:] * np.ones(D)[:,np.newaxis]) @ C
                U = map['U_raw'] * sigma_cluster[np.newaxis,np.newaxis,:,:] + map['U_raw_means'] @ C[np.newaxis,np.newaxis,:,:]
                V = map['V_raw'] * sigma_cluster[np.newaxis,np.newaxis,:,:] + map['V_raw_means'] @ C[np.newaxis,np.newaxis,:,:]
            elif self.include_clusters and self.ref_temp:
                sigma_temp = sigma_refT[:,np.newaxis,np.newaxis] * np.ones((D, N))[np.newaxis]
                sigma_temp = sigma_temp[:,np.newaxis,:,:]
                sigma_cluster = (np.sqrt(v_cluster)[np.newaxis,:] * np.ones(D)[:,np.newaxis]) @ C
                U = map['U_raw'] * sigma_temp + (map['U_raw_refT'] * sigma_cluster[np.newaxis,:,:] + map['U_raw_means'] @ C[np.newaxis,:,:])[np.newaxis,:,:,:]
                V = map['V_raw'] * sigma_temp + (map['U_raw_refT'] * sigma_cluster[np.newaxis,:,:] + map['V_raw_means'] @ C[np.newaxis,:,:])[np.newaxis,:,:,:]
            elif not self.include_clusters and self.ref_temp:
                sigma_temp = sigma_refT[:,np.newaxis,np.newaxis] * np.ones((D, N))[np.newaxis]
                sigma_temp = sigma_temp[:,np.newaxis,:,:]
                U = map['U_raw'] * sigma_temp + map['U_raw_refT'][np.newaxis,:,:,:]
                V = map['V_raw'] * sigma_temp + map['U_raw_refT'][np.newaxis,:,:,:]
            elif not self.include_clusters and not self.ref_temp:
                U = map['U_raw']
                V = map['V_raw']
            
            if self.ARD:
                v_ARD = np.diag(map['v_ARD'])[np.newaxis,np.newaxis,:,:]
            else:
                v_ARD = (100*np.eye(D))[np.newaxis,np.newaxis,:,:]

            A = U.transpose(0,1,3,2) @ v_ARD @ V
        
            a_rec += [np.column_stack([np.array([np.concatenate([np.concatenate([A[t,:,idx[0],idx[1]], A[t,:,idx[1],idx[0]][::-1]]) for t in range(N_T)]) for idx in Idx]).T,
                             np.array([np.concatenate([np.concatenate([A[t,:,idx[0],idx[1]], A[t,:,idx[1],idx[0]][::-1]]) for t in range(N_T)]) for idx in Idx]).T])]

        return np.stack(a_rec) # returns tensor of size len(MAP) x (N_T*N_C) x (N*(N-1)/2)

    def compute_testing_error(self):
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])
        N_C = json.load(open(self.data_file, 'r'))['N_C']
        N_T = json.load(open(self.data_file, 'r'))['N_T']
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = json.load(open(self.data_file, 'r'))['v_MC']

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

        idx1 = np.where(self.c_all[:, np.newaxis]==c1[idx][iidx][np.newaxis,:])[0]
        idx2 = np.where(self.c_all[:, np.newaxis]==c2[idx][iidx][np.newaxis,:])[0]

        testing_indices = np.column_stack([idx1, idx2])

        a_rec_testing = self.get_interps(Idx=testing_indices)

        y_MC = []

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])

        K_interp = self.K(x2, x2, T2, T2) + (jitter+v_MC) * np.eye(len(x2))
        L_inv_interp = np.linalg.inv(np.linalg.cholesky(K_interp))
        prec_interp = L_inv_interp.T @ L_inv_interp
        del K_interp, L_inv_interp

        for i in range(testing_indices.shape[0]):
            yy_MC = []
            mix1 = self.c_all[testing_indices[i,0]] + ' + ' + self.c_all[testing_indices[i,1]]
            idx = mix_all == mix1
            K_test_interp = self.K(data_dict['x'][idx], x2, data_dict['T'][idx], T2)
            
            for j in range(a_rec_testing.shape[0]):
                yy_MC += [K_test_interp @ prec_interp @ a_rec_testing[j,:,i]]
            
            yy_MC = np.stack(yy_MC, axis=0)
            y_MC += [yy_MC]

            del K_test_interp
        
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
        v_MC = json.load(open(self.data_file, 'r'))['v_MC']
        
        x2 = np.concatenate([x2_int for _ in range(N_T)])
        T2 = np.concatenate([T2_int[t]*np.ones_like(x2_int) for t in range(N_T)])

        data_dict = {'c1': np.concatenate([self.c_all[Idx_known[i,0]] for i in range(Idx_known.shape[0]) for j in range(N_points[i])]),
                     'c2': np.concatenate([self.c_all[Idx_known[i,1]] for i in range(Idx_known.shape[0]) for j in range(N_points[i])]),
                     'x': np.array(json.load(open(self.data_file, 'r'))['x1']),
                     'T': np.array(json.load(open(self.data_file, 'r'))['T1']),
                     'y_exp': np.array(json.load(open(self.data_file, 'r'))['y1'])}

        a_rec_testing = self.get_interps(Idx=Idx_known)

        y_MC = []

        K_interp = self.K(x2, x2, T2, T2) + (jitter+v_MC) * np.eye(len(x2))
        L_inv_interp = np.linalg.inv(np.linalg.cholesky(K_interp))
        prec_interp = L_inv_interp.T @ L_inv_interp
        del K_interp, L_inv_interp

        for i in range(Idx_known.shape[0]):
            yy_MC = []
            K_test_interp = self.K(data_dict['x'][np.sum(N_points[:i]):np.sum(N_points[:i+1])], x2, data_dict['T'][np.sum(N_points[:i]):np.sum(N_points[:i+1])], T2)
            
            for j in range(a_rec_testing.shape[0]):
                yy_MC += [K_test_interp @ prec_interp @ a_rec_testing[j,:,i]]
            
            yy_MC = np.stack(yy_MC, axis=0)
            y_MC += [yy_MC]

            del K_test_interp
        
        data_dict['y_MC'] = np.concatenate(y_MC, axis=1).T

        return data_dict
    
    def compute_error_metrics(self, data_dict=None):
        if data_dict is None:
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
                    ('UNIFAC', 'RMSE'): []}
        if data_dict is None:
            err_dict[('UNIFAC', 'MARE')] = []
        ranks = np.sort([int(i) for i in os.listdir(self.path) if i.isdigit()])
        for i in ranks:
            err_dict[f'MC ({i})', 'MAE'] = []
            err_dict[f'MC ({i})', 'RMSE'] = []
            if data_dict is None:
                err_dict[f'MC ({i})', 'MARE'] = []

        for j in range(len(unique_mix)):
            idx = mix_all == unique_mix[j]
            err_dict['Component 1', ''] += [data_dict['c1'][idx][0]]
            err_dict['Component 2', ''] += [data_dict['c2'][idx][0]]
            err_dict['UNIFAC', 'MAE'] += [np.mean(np.abs(y_exp[idx] - y_UNIFAC[idx]))]
            err_dict['UNIFAC', 'RMSE'] += [np.sqrt(np.mean((y_exp[idx] - y_UNIFAC[idx])**2))]
            if data_dict is None:
                err_dict['UNIFAC', 'MARE'] += [np.mean(np.abs((y_exp[idx] - y_UNIFAC[idx])/y_exp[idx]))]

            for i in ranks:
                err_dict[f'MC ({i})', 'MAE'] += [np.mean(np.abs(y_exp[idx] - y_MC[idx][:,i]))]
                err_dict[f'MC ({i})', 'RMSE'] += [np.sqrt(np.mean((y_exp[idx] - y_MC[idx][:,i])**2))]
                if data_dict is None:
                    err_dict[f'MC ({i})', 'MARE'] += [np.mean(np.abs((y_exp[idx] - y_MC[idx][:,i])/y_exp[idx]))]

        err_dict['Component 1', ''] += ['Overall']
        err_dict['Component 2', ''] += ['']
        err_dict['UNIFAC', 'MAE'] += [np.mean(np.abs(y_exp - y_UNIFAC))]
        err_dict['UNIFAC', 'RMSE'] += [np.sqrt(np.mean((y_exp - y_UNIFAC)**2))]
        if data_dict is None:
            err_dict['UNIFAC', 'MARE'] += [np.mean(np.abs((y_exp - y_UNIFAC)/y_exp))]
        for i in ranks:
            err_dict[f'MC ({i})', 'MAE'] += [np.mean(np.abs(y_exp - y_MC[:,i]))]
            err_dict[f'MC ({i})', 'RMSE'] += [np.sqrt(np.mean((y_exp - y_MC[:,i])**2))]
            if data_dict is None:
                err_dict[f'MC ({i})', 'MARE'] += [np.mean(np.abs((y_exp - y_MC[:,i])/y_exp))]

        return err_dict