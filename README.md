# Overview

The code listed in this reposistory is that developed in fullfilment of my masters' research project titled: **Prediction of excess enthalpy in binary mixture through probabilistic matrix completion** by *GR Hermanus (2024)*

# Background on Project
The topic deals with the use of probabilistic matrix factorization (PMF) for the prediction/imputation of excess enthalpy in Binary mixtures across both temperatures and composition. Two methods were proposed: 
1. **Pure Model**: Uses the excess enthalpy directly with a GP derived from a modified Redlich-Kister polynomial to ensure smoothness. PMF is applied to each combination of composition and temperature indepedently of one another, with the correlation of the GP enforcing correlation between the feature matrices at diferent conditions. This is not a valid Bayesian but rather a "typical" objective function
2. **Hybrid Model**: Uses the NRTL and predicts the NRTL parameters. PMF is applied on each of the NRTL parameters independently. These then in turn informs the excess enthalpy shapes. This model is a fully Bayesian model and MAP estimation along with sampling was performed.

The proposed models were compared to one another and to UNIFAC as a baseline.

# Dependencies
The code was constructed for application in Python. Several dependencies are required for the code to work.
1. Pandas: Processing the input data from excel sheets and converting to numpy arrays
2. [Stan](https://github.com/stan-dev/stan) and [cmdstanpy](https://github.com/stan-dev/cmdstanpy): Performing optimization and Bayesian inference
3. [Thermo](https://github.com/CalebBell/thermo): Prediction of excess enthalpy from UNIFAC
4. Matplotlib: Required for post processing plots

# Description of files and directory flow

## Pure model
The Pure model consisted of two tested implementtion. 
1. Pure RK PMF - No Temps: The Pure model implementation using data only at 298.15 K.
2. Pure RK PMF: The Pure model implementation using data across sourced temperatures.

## Hybrid model
The Hybrid model consisted of firstly obtaining the data-model mismatch parameters by performing Bayesian inference on the different training datasets independetly of one another.
1. Regression: Contains the Python, shell script and stan code to perform Bayesian inference to obtain the data model mismatch parameters. Code was also included to obtain the data-model mismatch parameters for the Pure models, these were however not used in the study due to large estimates for these values.
2. Hybrid PMF Adj: The Hybrid model implementation using data across soirced temperatures

## UNIFAC Predictions
The UNIFAC predictions were obtained by using the Thermo toolbox. Notebooks were written for the implementation of these
1. UNIFAC_EXCESS_PREDICTIONS.ipynb: Generate the predictions for the known datasets (testing and training)
2. UNIFAC_EXCESS_PREDICTIONS_UNKNOWN.ipynb: Generate the predictions for the unknown datasets (no testing nor training data)
3. UNIFAC_Plots.xlsx: UNIFAC predictions for plotting testing and training data
4. Thermo_UNIFAC_DMD_unknown.xlsx: UNIFAC predictions for unknown mixtures

## Data files
1. All Data.xlsx: Contains the orginally sourced (unsorted) data. This sheet also contains the functional group and cluster assignments for the different compounds considered.
2. Sorted Data.xlsx: Contains the training data sorted based on the compounds listed in All Data. Serveral other sheets are included to extract only training data for mixtures with more than 3 datapoints. This sheet also contains the UNIFAC predictions associated with the training mixtures at the conditions reported.
3. All_code.py: Contains the class "subsets" which extracts the subset of data from Sorted Data.xlsx corresponding to the functional groups given as input. These functions returns several additional infromation which can be used to extract the training data of interest, i.e. excluding those mixtures only consisting of 3 or less datapoints.
4. TestingData.xlsx: Contains the originally sourced training data
5. TestingData_Final.xlsx: Contains the testing data along with the UNIFAC predictions at the conditions reported
6. Sparsity.xlsx: Sheet showing the sparsity of the datasets collected in terms of compounds

**No claim is made on any of the excess enthalpy data sourced.** The data was sourced from across journal articles. The reference from which the data was sourced is listed in TestingRefs.xlsx and TrainingRefs.xlsx

## Post-Processing files
This code is dependent on the post-processing files in the Pure RK PMF, Pure RK PMF - No Temps and Hybrid PMF Adj directories depending on which script is being ran.
1. Comparison_of_Pure_and_Hybrid_Models.ipynb: Code for plotting graphs which shows comparisons between the Pure and Hybrid model
2. Overall_metrics.ipynb: Plots the metrics for the testing data for the different models implemented. Code needs to be changed to take into accout the different models
3. Plot_sparsity_plots_Pure_T_dep_vs_indep.ipynb: Code for plotting comparisons between the temperature dependent and independent models.
4. Pure_PMF_visualization_of_tensor_formation.ipynb: Code to plot graphs indicating how the matrices are formed from the testig data in the Pure model

## Depreciated/Not used
1. Hybrid PMF: Was an attempt at hyperparameter optimization through the incorporation of priors in a Bayesian framework and the usage of the ARD effect. These did not work as intended and should hence be ignored
2. k_mean.py: Was intended to perform clustering for the Pure compounds using the densities and boiling temperature at normal conditions, along with the molecular weight. Using these properties, the cluster assignments were not reflective of the clusters that were expected and was hence excluded in the study.




