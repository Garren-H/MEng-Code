This branch contains the code used for the Pure temperature independent model.
This model uses a kernel developed from the modified Redlich-Kister polynomial
and contraints excess enthalpy predictions to be smooth both wrt composition 
and temperature. The experimental excess enthalpy data is effectively interpolated 
using such a framework allowing for the incorporation of the data across compositions
and temperatures. 

The code was contructed with for computations on a clusters, hence the addition of
shell (.sh) scripts
