import cmdstanpy
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import json
from IPython.display import clear_output

sys.path.insert(0,'/home/garren/HPC Files')

from All_code import subsets # type: ignore

# Use Agg for saving a lot of plots without opening figure window
matplotlib.use('Agg')

#initialize by closing all open figures
plt.clf()
plt.close()

class Post_process:
    def __init__(self, include_clusters: bool, include_zeros: bool, ARD: bool, 
                 functional_groups: np.array, inf_type:str):
        
        self.include_clusters = include_clusters    # Add clusters
        self.include_zeros = include_zeros          # Add zeros         
        self.ARD = ARD                              # ARD
        self.functional_groups = functional_groups  # Functional groups
        self.inf_type = inf_type                    # Inference type

        self.path = '/home/garren/HPC Files/Pure RK PMF - No Temps/Subsets' # path to where files are stored
        self.path += f'/{self.functional_groups[0]}'
        for func in self.functional_groups[1:]:
            self.path += f'_{func}'
        self.data_file = f'{self.path}/data.json'                # data file
        self.path += f'/Include_clusters_{self.include_clusters}/Add_zeros_{self.include_zeros}/ARD_{self.ARD}/{self.inf_type}'

        # get all ranks
        self.ranks = [i for i in os.listdir(self.path) if i.isdigit()] # list of all ranks
        self.ranks = np.array(self.ranks).astype(int)
        self.ranks = np.sort(self.ranks)

        # save all compounds nanes in cluster along with functional group assignments
        with pd.ExcelFile('/home/garren/Documents/MEng/Code/Latest_results/HPC Files/All Data.xlsx') as f:
            self.fg = pd.read_excel(f, sheet_name='Pure compounds')['Functional Group'].to_numpy().astype(str)
            self.c_all = pd.read_excel(f, sheet_name='Pure compounds')['IUPAC'].to_numpy().astype(str)
            if functional_groups[0] != 'all':
                idx = np.sum(self.fg[:,np.newaxis] == functional_groups[np.newaxis,:], axis=1) > 0
                self.fg = self.fg[idx]
                self.c_all = self.c_all[idx]
                del idx

        # Kernel functions
        self.Kx = lambda x1, x2: np.column_stack([np.column_stack([x1**(i+2)-x1 for i in range(3)]), 
                                     1e-1*x1*np.sqrt(1-x1)*np.exp(x1)]) @ np.column_stack([np.column_stack([x2**(i+2)-x2 for i in range(3)]), 
                                                                                           1e-1*x2*np.sqrt(1-x2)*np.exp(x2)]).T
        self.KT = lambda T1, T2: np.column_stack([np.ones_like(T1), T1, T1**2, 1e-3*T1**3]) @ np.column_stack([np.ones_like(T2), T2, T2**2, 1e-3*T2**3]).T
        self.K = lambda x1, x2, T1, T2: self.Kx(x1, x2) * self.KT(T1, T2)

        # paths to excel files
        self.excel_testing = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/TestingData_Final.xlsx' # Path to testing data file
        self.excel_training = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Sorted Data.xlsx' # Path to training data file
        self.excel_plots_known = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/UNIFAC_Plots.xlsx' # Path to known data plots
        self.excel_unknown_vs_uni = '/home/garren/Documents/MEng/Code/Latest_results/HPC Files/Thermo_UNIFAC_DMD_unknown.xlsx' # Path to unknown data plots

        # Training Indices
        self.Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known'])-1

        # Testing Indices
        data_df = pd.read_excel(self.excel_testing)
        c1 = data_df['Component 1'].to_numpy().astype(str)
        c2 = data_df['Component 2'].to_numpy().astype(str)
        T_idx = np.abs(data_df['Temperature [K]'].to_numpy().astype(float) - 298.15) <= 0.5 

        del data_df

        idx1 = np.sum(c1[:, np.newaxis] == self.c_all[np.newaxis,:], axis=1)
        idx2 = np.sum(c2[:, np.newaxis] == self.c_all[np.newaxis,:], axis=1)

        idx = (idx1 + idx2) == 2
        idx = (idx.astype(int) + T_idx.astype(int)) == 2

        _, iidx = np.unique(np.char.add(np.char.add(c1[idx], ' + '), c2[idx]), return_index=True)
        iidx = np.sort(iidx)

        N = json.load(open(self.data_file, 'r'))['N']

        idx1 = np.sum((np.arange(N)[:, np.newaxis] * (self.c_all[:, np.newaxis]==c1[idx][iidx][np.newaxis,:])), axis=0)
        idx2 = np.sum((np.arange(N)[:, np.newaxis] * (self.c_all[:, np.newaxis]==c2[idx][iidx][np.newaxis,:])), axis=0)

        self.testing_indices = np.column_stack([idx1, idx2])

    def get_tensors(self):
        self.log_prob = []
        self.log_obj = []
        # set sran models for log_obj calculation
        stan_path = '/home/garren/HPC Files/Pure RK PMF - No Temps/Stan Models'
        stan_file = f'{stan_path}/Pure_PMF_include_clusters_{self.include_clusters}_zeros_{self.include_zeros}_ARD_{self.ARD}.stan'
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        A = [] # list of all A tensors to be converted to strings
        
        v_cluster = np.array(json.load(open(self.data_file, 'r'))['v_cluster'])
        C = np.array(json.load(open(self.data_file, 'r'))['C'])

        for rank in self.ranks:
            m = {}
            try:
                csv_file = [f'{self.path}/{rank}/{f}' for f in os.listdir(f'{self.path}/{rank}') if f.endswith('.csv')][0]
                MAP = cmdstanpy.from_csv(csv_file)
                keys = list(MAP.stan_variables().keys())
                for key in keys:
                    m[key] = MAP.stan_variables()[key]
                del MAP, csv_file
            except:
                inits_file = f'{self.path}/{rank}/inits.json'
                m = json.load(open(inits_file, 'r'))
                keys = list(m.keys())
                for key in keys:
                    m[key] = np.array(m[key])
                del inits_file
            data = json.load(open(self.data_file, 'r'))
            data['D'] = rank
            data['v_ARD'] = 100*np.ones(rank)
            self.log_prob += [model.log_prob(data=data, params=m).iloc[0,0]]
            self.log_obj += [np.log(-self.log_prob[-1])]

            sigma_cluster = np.sqrt(v_cluster)[np.newaxis,:] * np.ones((rank,1))
            sigma_cluster_mat = sigma_cluster @ C

            if self.include_clusters:
                U = m["U_raw"] * sigma_cluster_mat[np.newaxis,:,:] + m["U_raw_means"] @ C[np.newaxis,:,:]
                V = m["V_raw"] * sigma_cluster_mat[np.newaxis,:,:] + m["V_raw_means"] @ C[np.newaxis:,:]
            else:
                U = m["U_raw"]
                V = m["V_raw"]

            if self.ARD:
                v_ARD = np.diag(m["v_ARD"])[np.newaxis,:,:]
            else:
                v_ARD = 100*np.eye(rank)[np.newaxis,:,:]
            
            A += [U.transpose(0,2,1) @ v_ARD @ V]
        
        return np.array(A)
    
    def extract_interps(self, Idx: np.array, A=None):
        if A is None:
            A = self.get_tensors()
        R = A.shape[0]
        M = A.shape[1]
        N = A.shape[2]
        y_MC_pred = np.stack([np.column_stack([np.concatenate([A[r,:, idx[0], idx[1]], A[r,::-1,idx[1],idx[0]]])
                            for idx in Idx]) for r in range(R)])
        
        return y_MC_pred
    
    def get_reconstructed_values(self, A=None):
        if A is None:
            A = self.get_tensors()

        y_MC_interp = self.extract_interps(A=A, Idx=self.Idx_known)

        del A

        N_known = json.load(open(self.data_file, 'r'))['N_known']
        N_points = np.array(json.load(open(self.data_file, 'r'))['N_points'])
        x = np.array(json.load(open(self.data_file, 'r'))['x1'])
        T = np.array(json.load(open(self.data_file, 'r'))['T1'])
        v = np.array(json.load(open(self.data_file, 'r'))['v'])


        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])

        x_MC = x2_int
        T_MC = T2_int*np.ones_like(x2_int)
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = np.array(json.load(open(self.data_file, 'r'))['v_MC'])

        K_MC = self.K(x_MC, x_MC, T_MC, T_MC) + (jitter+v_MC)*np.eye(x_MC.shape[0])
        L_MC = np.linalg.cholesky(K_MC)
        L_MC_inv = np.linalg.inv(L_MC)
        K_MC_inv = L_MC_inv.T @ L_MC_inv
        del K_MC, L_MC

        y_MC = []
        for i in range(N_known):
            xx = x[np.sum(N_points[:i]):np.sum(N_points[:i+1])]
            TT = T[np.sum(N_points[:i]):np.sum(N_points[:i+1])]
            K_y_MC = self.K(xx, x_MC, TT, T_MC)

            y_MC += [K_y_MC @ K_MC_inv @ y_MC_interp[:,:,i].T]

        y_MC = np.concatenate(y_MC, axis=0)

        data_dict = {'c1': [],
                     'c2': [],
                     'x': x,
                     'T': T,
                     'y_exp': np.array(json.load(open(self.data_file, 'r'))['y1']),
                     'y_MC': y_MC}
        
        # Add compound names
        data_dict['c1'] = np.concatenate([[self.c_all[self.Idx_known[i,0]]]*N_points[i] for i in range(N_known)])
        data_dict['c2'] = np.concatenate([[self.c_all[self.Idx_known[i,1]]]*N_points[i] for i in range(N_known)])
        
        # Extract UNIFAC Data
        subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T = subsets(self.functional_groups).get_subset_df()
        y_UNIFAC = np.concatenate([subset_df['UNIFAC_DMD [J/mol]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
        T_UNIFAC = np.concatenate([subset_df['Temperature [K]'][subset_Indices_T[j,0]:subset_Indices_T[j,1]+1].to_numpy() for j in range(subset_Indices_T.shape[0])])
        T_idx = np.abs(T_UNIFAC - 298.15) <= 0.5
        del subset_df, subset_Indices, subset_Indices_T, Info_Indices, init_indices, init_indices_T

        data_dict['y_UNIFAC'] = y_UNIFAC[T_idx]

        return data_dict
    
    def get_testing_values(self, A=None):
        # Get matrices
        if A is None:
            A = self.get_tensors()

        # Extraxt testing data from excel
        data_df = pd.read_excel(self.excel_testing)

        T = data_df['Temperature [K]'].to_numpy().astype(float)
        T_idx = np.abs(T - 298.15) <= 0.5
        T = 298.15 * np.ones((np.sum(T_idx)))
        x = data_df['Composition component 1 [mol/mol]'].to_numpy().astype(float)[T_idx]
        y_exp = data_df['Excess Enthalpy [J/mol]'].to_numpy().astype(float)[T_idx]
        y_UNIFAC = data_df['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[T_idx]
        c1 = data_df['Component 1'].to_numpy().astype(str)[T_idx]
        c2 = data_df['Component 2'].to_numpy().astype(str)[T_idx]

        del data_df

        idx1 = np.sum(c1[:, np.newaxis] == self.c_all[np.newaxis,:], axis=1)
        idx2 = np.sum(c2[:, np.newaxis] == self.c_all[np.newaxis,:], axis=1)

        idx = (idx1 + idx2) == 2

        data_dict = {'c1': c1[idx],
                     'c2': c2[idx],
                     'x': x[idx],
                     'T': T[idx],
                     'y_exp': y_exp[idx],
                     'y_UNIFAC': y_UNIFAC[idx],}

        # get interpolated values
        y_MC_interp = self.extract_interps(A=A, Idx=self.testing_indices)

        # All mixtures
        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])

        # Extract data from file
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])

        x_MC = x2_int
        T_MC = T2_int*np.ones_like(x2_int)
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = np.array(json.load(open(self.data_file, 'r'))['v_MC'])

        K_MC = self.K(x_MC, x_MC, T_MC, T_MC) + (jitter+v_MC)*np.eye(x_MC.shape[0])
        L_MC = np.linalg.cholesky(K_MC)
        L_MC_inv = np.linalg.inv(L_MC)
        K_MC_inv = L_MC_inv.T @ L_MC_inv
        del K_MC, L_MC

        y_MC = []
        for i in range(self.testing_indices.shape[0]):
            mix1 = self.c_all[self.testing_indices[i,0]] + ' + ' + self.c_all[self.testing_indices[i,1]]
            idx = mix_all == mix1

            xx = data_dict['x'][idx]
            TT = data_dict['T'][idx]
            K_y_MC = self.K(xx, x_MC, TT, T_MC)

            y_MC += [K_y_MC @ K_MC_inv @ y_MC_interp[:,:,i].T]
        
        y_MC = np.concatenate(y_MC, axis=0)

        data_dict['y_MC'] = y_MC

        return data_dict
    
    def get_testing_metrics(self, A=None):
        if A is None:
            A = self.get_tensors()
        data_dict = self.get_testing_values(A=A)

        mix_all = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        err_dict = {('Component 1', '', ''): [],
                    ('Component 2', '', ''): [],
                    ('UNIFAC', 'MAE', ''): [],
                    ('UNIFAC', 'RMSE', ''): [],
                    ('UNIFAC', 'MARE', ''): [],}
        metrics = ['MAE', 'RMSE', 'MARE']
        for metric in metrics:
            for m in range(len(self.ranks)):
                err_dict[('MC', metric, self.ranks[m])] = []

        y_exp = data_dict['y_exp']
        y_UNIFAC = data_dict['y_UNIFAC']
        y_MC = data_dict['y_MC']

        for j in range(len(unique_mix)):
            idx = mix_all == unique_mix[j]
            err_dict['Component 1', '', ''] += [data_dict['c1'][idx][0]]
            err_dict['Component 2', '', ''] += [data_dict['c2'][idx][0]]
            err_dict['UNIFAC', 'MAE', ''] += [np.mean(np.abs(y_exp[idx] - y_UNIFAC[idx]))]
            err_dict['UNIFAC', 'RMSE', ''] += [np.sqrt(np.mean((y_exp[idx] - y_UNIFAC[idx])**2))]
            err_dict['UNIFAC', 'MARE', ''] += [np.mean(np.abs((y_exp[idx] - y_UNIFAC[idx])/y_exp[idx]))]
            for m in range(len(self.ranks)):
                err_dict['MC', 'MAE', self.ranks[m]] += [np.mean(np.abs(y_exp[idx] - y_MC[idx,m]))]
                err_dict['MC', 'RMSE', self.ranks[m]] += [np.sqrt(np.mean((y_exp[idx] - y_MC[idx,m])**2))]
                err_dict['MC', 'MARE', self.ranks[m]] += [np.mean(np.abs((y_exp[idx] - y_MC[idx,m])/y_exp[idx]))]

        err_dict['Component 1', '', ''] += ['Overall']
        err_dict['Component 2', '', ''] += ['']
        err_dict['UNIFAC', 'MAE', ''] += [np.mean(np.abs(y_exp - y_UNIFAC))]
        err_dict['UNIFAC', 'RMSE', ''] += [np.sqrt(np.mean((y_exp - y_UNIFAC)**2))]
        err_dict['UNIFAC', 'MARE', ''] += [np.mean(np.abs((y_exp - y_UNIFAC)/y_exp))]
        for m in range(len(self.ranks)):
            err_dict['MC', 'MAE', self.ranks[m]] += [np.mean(np.abs(y_exp - y_MC[:,m]))]
            err_dict['MC', 'RMSE', self.ranks[m]] += [np.sqrt(np.mean((y_exp - y_MC[:,m])**2))]
            err_dict['MC', 'MARE', self.ranks[m]] += [np.mean(np.abs((y_exp - y_MC[:,m])/y_exp))]

        keys = list(err_dict.keys())
        for key in keys:
            err_dict[key] = np.array(err_dict[key])

        return err_dict
    
    def plot_err_metrics(self, A=None):
        if A is None:
            A = self.get_tensors()
        err_dict = self.get_testing_metrics(A=A)

        MAE = [err_dict['MC', 'MAE', r][-1] for r in self.ranks]
        RMSE = [err_dict['MC', 'RMSE', r][-1] for r in self.ranks]
        MARE = [err_dict['MC', 'MARE', r][-1]*100 for r in self.ranks]

        p1 = plt.plot(self.ranks, MAE, '.k', label='MAE', alpha=0.5)
        p2 = plt.plot(self.ranks, RMSE, '.r', label='RMSE', alpha=0.5)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.ylabel('MAE, RMSE [J/mol]', fontsize=15)
        plt.xlabel('Rank', fontsize=15)
        plt.xticks(self.ranks, self.ranks.astype(str))
        p3 = plt.twinx().plot(self.ranks, MARE, '.g', label='MARE', alpha=0.5)
        p4 = plt.plot(self.ranks, self.log_obj, '.b', label='Log Objective', alpha=0.5)
        plt.ylabel('MARE [%], Log Objective [-]', fontsize=15)

        p = p1+p2+p3+p4
        labs = [l.get_label() for l in p]
        plt.legend(p, labs, loc='upper left', bbox_to_anchor=(1.2, 1))
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()
        png_path = f'{self.path}/Overall_error_metrics.png'

        plt.savefig(png_path)
        plt.clf()
        plt.close()

    def plot_predicted_vs_experimental(self, data_type=None, A=None):
        if A is None:
            A = self.get_tensors()
        if data_type == 'Testing':
            data_dict = self.get_testing_values(A=A)
        elif data_type == 'Training':
            data_dict = self.get_reconstructed_values(A=A)
        else:
            print('Please specify data type')
            return
        
        for m in range(len(self.ranks)):
            plt.plot(data_dict['y_UNIFAC'], data_dict['y_exp'], '.g', label='UNIFAC', markersize=3)
            plt.plot(data_dict['y_MC'][:,m], data_dict['y_exp'], '.r', label=f'MC Rank {self.ranks[m]}', markersize=3)
            min_lim = np.min([plt.xlim()[0], plt.ylim()[0]])
            max_lim = np.max([plt.xlim()[1], plt.ylim()[1]])
            plt.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')
            plt.legend()
            plt.xlabel('Predicted Excess Enthalpy [J/mol]', fontsize=15)
            plt.ylabel('Experimental Excess Enthalpy [J/mol]', fontsize=15)
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.tight_layout()
            png_path = f'{self.path}/{data_type}_Predicted_vs_experimental_rank_{self.ranks[m]}.png'
            plt.savefig(png_path, dpi=300)
            plt.clf()
            plt.close()

    def plot_predicted_hist(self, data_type='None', A=None):
        if A is None:
            A = self.get_tensors()
        if data_type == 'Testing':
            data_dict = self.get_testing_values(A=A)
        elif data_type == 'Training':
            data_dict = self.get_reconstructed_values(A=A)
        else:
            print('Please specify data type')
            return

        for m in range(len(self.ranks)):
            A_MC = np.abs(data_dict['y_MC'][:,m] - data_dict['y_exp'])
            A_UNIFAC = np.abs(data_dict['y_UNIFAC'] - data_dict['y_exp'])

            max_val = 1000
            bins = np.linspace(0, max_val, 101)
            bins = np.append(bins, bins[-1]+bins[1]-bins[0])
            x_ticks = np.array([0, 200, 400, 600, 800, 1000])
            x_tick_labels = x_ticks.astype(str)
            x_tick_labels[-1] += '+'

            A_MC[A_MC > max_val] = max_val+10
            A_UNIFAC[A_UNIFAC > max_val] = max_val+10

            plt.hist(A_MC, bins=bins, alpha=0.5, label=f'MC Rank {self.ranks[m]}', color='r')
            plt.hist(A_UNIFAC, bins=bins, alpha=0.5, label='UNIFAC', color='g')
            plt.legend()
            plt.xlabel('Absolute Error [J/mol]', fontsize=15)
            plt.ylabel('Frequency', fontsize=15)
            plt.xlim([bins[0], bins[-1]])
            plt.xticks(x_ticks, x_tick_labels)
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.tight_layout()
            png_path = f'{self.path}/{data_type}_Histogram_rank_{self.ranks[m]}.png'
            plt.savefig(png_path, dpi=300)
            plt.clf()
            plt.close()

    def plot_functional_groups_MC_vs_UNIFAC(self, A=None):
        
        if A is None:
            A = self.get_tensors()
        err_metrics = ['RMSE', 'MAE', 'MARE']
        N = json.load(open(self.data_file, 'r'))['N']
        AE_MC = np.nan*np.eye(N)
        AE_uni = np.nan*np.eye(N)

        err_dict = self.get_testing_metrics(A=A)

        for err_metric in err_metrics:
            for m in range(len(self.ranks)):
                if err_metric == 'MARE':
                    cutoff = 0.2
                else:
                    cutoff = 50

                diff_metrics = np.array(err_dict['MC', err_metric, self.ranks[m]][:-1]) - np.array(err_dict['UNIFAC', err_metric, ''][:-1])

                uni_best = diff_metrics>0
                MC_best = ~uni_best

                diff_metrics = np.abs(diff_metrics)
                diff_metrics[diff_metrics <= cutoff] = 0.2
                diff_metrics[diff_metrics > cutoff] = 1

                AE_MC[self.testing_indices[MC_best,0], self.testing_indices[MC_best,1]] = diff_metrics[MC_best]
                AE_uni[self.testing_indices[uni_best,0], self.testing_indices[uni_best,1]] = diff_metrics[uni_best]

                plt.figure(figsize=(10, 10))

                plt.plot(self.Idx_known[:,1], self.Idx_known[:,0], '*k', label='Training Data', alpha=0.5, markersize=3)
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=12)

                plt.imshow(AE_MC, cmap='Reds', vmin=0, vmax=1, label='MC')
                plt.imshow(AE_uni, cmap='Greens', vmin=0, vmax=1, label='UNIFAC')

                A_grey = np.nan*np.eye(N)
                for i in range(N):
                    for j in range(i,N):
                        A_grey[j,i] = 0.25
                    
                plt.imshow(A_grey, cmap='Greys',vmin=0,vmax=1)

                unique_fg, idx, counts = np.unique(self.fg, return_index=True, return_counts=True)
                unique_fg = unique_fg[np.argsort(idx)]
                counts = counts[np.argsort(idx)]
                counts[0]=counts[0]-1
                counts = counts

                end_points = [0]
                for count in np.cumsum(counts):
                    count += 0.5
                    end_points += [count]
                    plt.plot([count, count], [0, N-1], '--k', alpha=0.3)
                    plt.plot([0, N-1], [count, count], '--k', alpha=0.3)

                mid_points = (np.array(end_points[:-1])+np.array(end_points[1:]))/2
                plt.xticks(mid_points, unique_fg, rotation=90, fontsize=12)
                plt.yticks(mid_points, unique_fg, fontsize=12)

                plt.tight_layout()

                png_path = f'{self.path}/Testing_{err_metric}_rank_{self.ranks[m]}.png'

                plt.savefig(png_path, dpi=500)

                plt.clf()
                plt.close()

    def plot_functional_groups_MC_colorbar(self, A=None):
        if A is None:
            A = self.get_tensors()
        err_metrics = ['RMSE', 'MAE', 'MARE']
        N = json.load(open(self.data_file, 'r'))['N']
        AE_MC = np.nan*np.eye(N)

        err_dict = self.get_testing_metrics(A=A)

        for err_metric in err_metrics:
            if err_metric == 'MARE':
                levels = [1e-2, 1e-1, 1e-0, 1e1, 1e2]
            else:
                levels = [1e0, 1e1, 1e2, 1e3, 1e4]
            for m in range(len(self.ranks)):
                diff_metrics = err_dict['MC', err_metric, self.ranks[m]][:-1]

                AE_MC[self.testing_indices[:,0], self.testing_indices[:,1]] = diff_metrics

                plt.figure(figsize=(10, 10))

                plt.plot(self.Idx_known[:,1], self.Idx_known[:,0], '*k', label='Training Data', alpha=0.5, markersize=3)
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), fontsize=12)

                plt.imshow(AE_MC, cmap='viridis', label='MC', norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
                cbar = plt.colorbar()
                cbar.set_ticks(levels)

                A_grey = np.nan*np.eye(N)
                for i in range(N):
                    for j in range(i,N):
                        A_grey[j,i] = 0.25
                    
                plt.imshow(A_grey, cmap='Greys',vmin=0,vmax=1)

                unique_fg, idx, counts = np.unique(self.fg, return_index=True, return_counts=True)
                unique_fg = unique_fg[np.argsort(idx)]
                counts = counts[np.argsort(idx)]
                counts[0]=counts[0]-1
                counts = counts

                end_points = [0]
                for count in np.cumsum(counts):
                    count += 0.5
                    end_points += [count]
                    plt.plot([count, count], [0, N-1], '--k', alpha=0.3)
                    plt.plot([0, N-1], [count, count], '--k', alpha=0.3)

                mid_points = (np.array(end_points[:-1])+np.array(end_points[1:]))/2
                plt.xticks(mid_points, unique_fg, rotation=90, fontsize=12)
                plt.yticks(mid_points, unique_fg, fontsize=12)

                plt.tight_layout()

                png_path = f'{self.path}/Testing_{err_metric}_rank_{self.ranks[m]}_MC_only.png'

                plt.savefig(png_path, dpi=500)

                plt.clf()
                plt.close()

    def plot_2D_plots(self, data_type=None, ranks=None, plot_one=False, A=None):
        if A is None:
            A = self.get_tensors()
        
        if data_type == 'Testing':
            Idx = self.testing_indices
            y_MC_interp = self.extract_interps(A=A, Idx=Idx)
            data_dict = self.get_testing_values(A=A)
        elif data_type == 'Training':
            Idx = self.Idx_known
            y_MC_interp = self.extract_interps(A=A, Idx=Idx)
            data_dict = self.get_reconstructed_values(A=A)
        else:
            print('Please specify data type')
            return

        excel_UNI_sheet = f'{data_type}_Plots'
        
        if ranks is None:
            ranks = self.ranks
            ranks_idx = np.arange(len(ranks))
        else:
            ranks = np.array(ranks).astype(int)
            ranks_idx = np.zeros_like(self.ranks).astype(int)
            for r in ranks:
                ranks_idx += (self.ranks == r).astype(int)
            ranks_idx = np.where(ranks_idx.astype(bool))[0]
        
        for r in ranks: # test if rank is in ranks
            if np.sum(self.ranks == r) == 0:
                print(f'Rank {r} not found')
                return
        
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])
        x_MC = x2_int
        T_MC = T2_int*np.ones_like(x2_int)
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = np.array(json.load(open(self.data_file, 'r'))['v_MC'])

        K_MC = self.K(x_MC, x_MC, T_MC, T_MC) + (jitter+v_MC)*np.eye(x_MC.shape[0])
        L_MC = np.linalg.cholesky(K_MC)
        L_MC_inv = np.linalg.inv(L_MC)
        K_MC_inv = L_MC_inv.T @ L_MC_inv
        del K_MC, L_MC

        mix_all = np.char.add(np.char.add(self.c_all[Idx[:,0]], ' + '), self.c_all[Idx[:,1]])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]
        df_UNIFAC = pd.read_excel(self.excel_plots_known, sheet_name=excel_UNI_sheet)
        UNIFAC_mix = np.char.add(np.char.add(df_UNIFAC['Component 1'].to_numpy().astype(str), ' + '), df_UNIFAC['Component 2'].to_numpy().astype(str))

        exp_mix = np.char.add(np.char.add(data_dict['c1'], ' + '), data_dict['c2'])

        colours = ['r', 'b', 'magenta', 'y', 'saddlebrown', 'k', 'cyan', 'lime']

        if plot_one:
            png_path = f'{self.path}/{ranks[0]}/2D Plots/{data_type}'
        else:
            png_path = f'{self.path}/2D Plots/{data_type}'

        try:
            os.makedirs(png_path)
        except:
            pass

        for j in range(Idx.shape[0]):
            y_idx = exp_mix == unique_mix[j]
            UNIFAC_idx = UNIFAC_mix == unique_mix[j]
            yy = data_dict['y_exp'][y_idx]
            yy_UNIFAC = df_UNIFAC['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[UNIFAC_idx]
            x_y = data_dict['x'][y_idx]
            T_y = data_dict['T'][y_idx]
            c1 = self.c_all[Idx[j,0]]
            c2 = self.c_all[Idx[j,1]]
            x_UNIFAC = df_UNIFAC['Composition component 1 [mol/mol]'].to_numpy().astype(float)[UNIFAC_idx]
            T_UNIFAC = df_UNIFAC['Temperature [K]'].to_numpy().astype(float)[UNIFAC_idx]

            K_pred_MC = self.K(x_UNIFAC, x_MC, T_UNIFAC, T_MC)

            if plot_one:
                yy_MC_mean = K_pred_MC @ K_MC_inv @ y_MC_interp[ranks_idx,:,j][0,:]    
            else:
                yy_MC_mean = K_pred_MC @ K_MC_inv @ y_MC_interp[ranks_idx,:,j].T

            T_uniq = np.unique(T_y)
            T_uniq = np.concatenate([T_uniq.astype(int)+0.15,T_uniq.astype(int)+1.15])
            T_uniq = np.unique(T_uniq)
            T_uniq = T_uniq[np.sum(np.abs(T_uniq[:,np.newaxis]-T_y[np.newaxis,:]) <= 0.5, axis=1) > 0]
            for i in range(len(T_uniq)):
                fig, ax = plt.subplots()
                TT = T_uniq[i]
                T_y_idx = np.abs(T_y - TT) <= 0.5
                T_UNIFAC_idx = T_UNIFAC == TT

                if plot_one:
                    ax.plot(x2_int, y_MC_interp[ranks_idx,:,j][0,:], '.r', markersize=15, label=f'MC Rec Rank {ranks[0]}')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx], '-r', label=f'MC Smooth Rank {ranks[0]}')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_UNIFAC[T_UNIFAC_idx], '-g', label='UNIFAC')
                else:
                    for m in range(len(ranks)):
                        ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx,m], '-', color=f'{colours[ranks_idx[m]]}', label=f'MC Rank {ranks[m]}')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_UNIFAC[T_UNIFAC_idx], '--g', label='UNIFAC')
                ax.set_xlabel('Composition of Compound 1 [mol/mol]', fontsize=15)
                ax.set_ylabel('Excess Enthalpy [J/mol]', fontsize=18)
                ax.set_title(f'(1) {c1} + (2) {c2} at {T_uniq[i]:.2f} K', fontsize=13)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.plot(x_y[T_y_idx], yy[T_y_idx], '.k', label='Experimental Data', markersize=15)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.tight_layout()

                fig_path = f'{png_path}/{j}_{i}.png'
                fig.savefig(fig_path, dpi=300)
                plt.clf()
                plt.close()

                clear_output(wait=False)
                print(f'{j}_{i}')
        clear_output(wait=False)
      
    def plot_functional_groups_Unknown(self, A=None):
        if A is None:
            A = self.get_tensors()
        UNIFAC_df = pd.read_excel(self.excel_unknown_vs_uni)
        T_idx = np.abs(UNIFAC_df['Temperature [K]'].to_numpy().astype(float) - 298.15) <= 0.5
        UNIFAC_mix = np.char.add(np.char.add(UNIFAC_df['Component 1'].to_numpy().astype(str), ' + '), UNIFAC_df['Component 2'].to_numpy().astype(str))[T_idx]
        UNIFAC_y = UNIFAC_df['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[T_idx]
        del UNIFAC_df

        N = json.load(open(self.data_file, 'r'))['N']
        Idx_unknown_all = np.array([[i, j] for i in range(N) for j in range(i+1, N)])
        Idx_test_and_train = np.concatenate([self.Idx_known, self.testing_indices])

        idx_keep = np.sum(np.char.add(np.char.add(Idx_unknown_all[:,0].astype(str), ' + '), 
                           Idx_unknown_all[:,1].astype(str))[:,np.newaxis] == np.char.add(np.char.add(Idx_test_and_train[:,0].astype(str), ' + '), 
                                                                                          Idx_test_and_train[:,1].astype(str))[np.newaxis,:], axis=1) == 0
        
        Idx_unknown = Idx_unknown_all[idx_keep]
        y_MC_unknown_all = self.extract_interps(A=A, Idx=Idx_unknown)

        c1 = self.c_all[Idx_unknown[:,0]]
        c2 = self.c_all[Idx_unknown[:,1]]
        MC_mix = np.char.add(np.char.add(c1, ' + '), c2)

        N_C = json.load(open(self.data_file, 'r'))['N_C']
        N_points = N_C

        RMSE = []
        MAE = []
        for j in range(Idx_unknown.shape[0]):
            idx = UNIFAC_mix == MC_mix[j]

            RMSE += [np.sqrt(np.mean((UNIFAC_y[idx][np.newaxis,:] - y_MC_unknown_all[:,:,j])**2, axis=1))]
            MAE += [np.mean(np.abs(UNIFAC_y[idx][np.newaxis,:] - y_MC_unknown_all[:,:,j]), axis=1)]
        
        RMSE = np.array(RMSE)
        MAE = np.array(MAE)

        err_metrics = ['RMSE', 'MAE']
        N = json.load(open(self.data_file, 'r'))['N']
        AE_MC = np.nan*np.eye(N)

        err_dict = {'RMSE': RMSE,
                    'MAE': MAE}

        for err_metric in err_metrics:
            levels = [1e0, 1e1, 1e2, 1e3, 1e4]
            for m in range(len(self.ranks)):
                diff_metrics = err_dict[err_metric][:,m]

                AE_MC[Idx_unknown[:,0], Idx_unknown[:,1]] = diff_metrics

                plt.figure(figsize=(10, 10))

                plt.plot(self.Idx_known[:,1], self.Idx_known[:,0], '*k', label='Training Data', alpha=0.5, markersize=3)
                plt.plot(self.testing_indices[:,1], self.testing_indices[:,0], '*r', label='Testing Data', alpha=0.5, markersize=3)
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=12)

                plt.imshow(AE_MC, cmap='viridis', label='MC', norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
                cbar = plt.colorbar()
                cbar.set_ticks(levels)

                A_grey = np.nan*np.eye(N)
                for i in range(N):
                    for j in range(i,N):
                        A_grey[j,i] = 0.25
                    
                plt.imshow(A_grey, cmap='Greys',vmin=0,vmax=1)

                unique_fg, idx, counts = np.unique(self.fg, return_index=True, return_counts=True)
                unique_fg = unique_fg[np.argsort(idx)]
                counts = counts[np.argsort(idx)]
                counts[0]=counts[0]-1
                counts = counts

                end_points = [0]
                for count in np.cumsum(counts):
                    count += 0.5
                    end_points += [count]
                    plt.plot([count, count], [0, N-1], '--k', alpha=0.3)
                    plt.plot([0, N-1], [count, count], '--k', alpha=0.3)

                mid_points = (np.array(end_points[:-1])+np.array(end_points[1:]))/2
                plt.xticks(mid_points, unique_fg, rotation=90, fontsize=12)
                plt.yticks(mid_points, unique_fg, fontsize=12)

                plt.tight_layout()

                png_path = f'{self.path}/Unknown_{err_metric}_rank_{self.ranks[m]}.png'

                plt.savefig(png_path, dpi=500)

                plt.clf()
                plt.close()

        






            
    

        


