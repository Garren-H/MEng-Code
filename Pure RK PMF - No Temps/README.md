This branch contains the code used for the Pure temperature independent model.
This model uses a kernel developed from the modified Redlich-Kister polynomial
and contraints excess enthalpy predictions to be smooth both wrt composition 
and temperature. The experimental excess enthalpy data is effectively interpolated 
using such a framework allowing for the incorporation of the data across compositions
and temperatures. This is not a Bayesian model, but rather just a typical optimization 
problem derived from GP's. The stan programming language was hence only used as a method
of optimization in this case

The code was contructed with for computations on a clusters, hence the addition of
shell (.sh) scripts
