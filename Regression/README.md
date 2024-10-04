This branch was used to
1. Regress the NRTL parameters based on the excess enthalpy for every dataset
   independently of one another. From this framework the data-model mismatch
   parameters are obtained which is futher used in the Hybrid PMF Adj branch
2. Perform Bayesian inference to obtain the data-model mismatch from the modified
   Redlich-Kister kernel. This was supposed to be used in the Pure RK PMF frameworks
   however, these variances were frequently estimate to be large, and hence not included. 

This is a workflow for submitting the same python files to a PBS cluster. 
The workflow firstly checks how many <name>.json files there are in the Data folder
Where <name> is an integer value for each of the mixtures being considered, starting at 0. 
The file workflow checks if there is running jobs currently with the name "Regression_<name>"
If there is it writes these to a .txt file. It then checks if there are completed datsets and
Stores this is a different .txt file. If the <name> is not in any of these .txt files it is
written to a different .txt file which is used for the automated queing of jobs.
The files having _Step3 in the name is that for the final step which is should be ran after
the initialization files, without _Step3 extension
