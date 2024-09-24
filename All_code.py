import numpy as np #type: ignore 
import pandas as pd #type:ignore

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



        








        


            
            





        



