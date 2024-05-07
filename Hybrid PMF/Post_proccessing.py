'''
Python file for post-processing of results for Hybrid Model
'''

import numpy as np

def get_Idx_unknown(N, Idx_known):
    Idx_unknown = np.array([[i, j] for i in range(N) for j in range(i+1,N)])
    idx = np.sum(np.char.add(Idx_unknown[:,0].astype(str), Idx_unknown[:,1].astype(str))[:,np.newaxis] == np.char.add(Idx_known[:,0].astype(str), Idx_known[:,1].astype(str))[np.newaxis,:], axis=1) == 0
    Idx_unknown = Idx_unknown[idx]

    return Idx_unknown

def get_known_and_unknown_params(U_raw, v_ARD, V_raw, Idx_known, Idx_unknown, scaling):
    A = U_raw.transpose(0,1,3,2) @ (v_ARD[:,np.newaxis,:] * np.eye(v_ARD.shape[1])[np.newaxis,:,:])[:,np.newaxis,:,:] @ V_raw
    p12_known = A[:,:,Idx_known[:,0], Idx_known[:,1]] * scaling[np.newaxis,:,np.newaxis]
    p21_known = A[:,:,Idx_known[:,1], Idx_known[:,0]] * scaling[np.newaxis,:,np.newaxis]
    p12_unknown = A[:,:,Idx_unknown[:,0], Idx_unknown[:,1]] * scaling[np.newaxis,:,np.newaxis]
    p21_unknown = A[:,:,Idx_unknown[:,1], Idx_unknown[:,0]] * scaling[np.newaxis,:,np.newaxis]
    
    return p12_known, p21_known, p12_unknown, p21_unknown

def excess_enthalpy_predictions(x, T, p12, p21, a=0.3):
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

