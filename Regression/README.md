### This is a workflow for submitting the same python files to a PBS cluster. 
The workflow firstly checks how many <name>.json files there are in the Data folder
Where <name> is an integer value for each of the mixtures being considered, starting at 0. 
The file workflow checks if there is running jobs currently with the name "Regression_<name>"
If there is it writes these to a .txt file. It then checks if there are completed datsets and
Stores this is a different .txt file. If the <name> is not in any of these .txt files it is
written to a different .txt file which is used for the automated queing of jobs.
The files having _Step3 in the name is that for the final step which is should be ran after
the initialization files, without _Step3 extension
