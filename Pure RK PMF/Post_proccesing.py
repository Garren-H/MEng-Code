'''
Function file for postprocessing of results
'''

import numpy as np

M = int((N_C+1)/2)

#Point estimate reconstructed values
a_rec_known = np.concatenate([np.concatenate([np.concatenate([np.stack([(MAP.U_raw[t,m,:,:].T @ np.diag(MAP.v_ARD) @ MAP.V_raw[t,m,:,:])[Idx_known[:,0], Idx_known[:,1]] for m in range(M-1)], axis=0 ), 
                ((MAP.U_raw[t,-1,:,:].T @ np.diag(MAP.v_ARD) @ MAP.U_raw[t,-1,:,:])[Idx_known[:,0], Idx_known[:,1]][np.newaxis,:])], axis=0), 
                np.stack([(MAP.U_raw[t,m,:,:].T @ np.diag(MAP.v_ARD) @ MAP.V_raw[t,m,:,:])[Idx_known[:,1], Idx_known[:,0]] for m in range(M-1)], axis=0 )[::-1,:]], axis=0) for t in range(N_T)], axis=0)

a_rec_unknown = np.concatenate([np.concatenate([np.concatenate([np.stack([(MAP.U_raw[t,m,:,:].T @ np.diag(MAP.v_ARD) @ MAP.V_raw[t,m,:,:])[Idx_unknown[:,0], Idx_unknown[:,1]] for m in range(M-1)], axis=0 ), 
                ((MAP.U_raw[t,-1,:,:].T @ np.diag(MAP.v_ARD) @ MAP.U_raw[t,-1,:,:])[Idx_unknown[:,0], Idx_unknown[:,1]][np.newaxis,:])], axis=0), 
                np.stack([(MAP.U_raw[t,m,:,:].T @ np.diag(MAP.v_ARD) @ MAP.V_raw[t,m,:,:])[Idx_unknown[:,1], Idx_unknown[:,0]] for m in range(M-1)], axis=0 )[::-1,:]], axis=0) for t in range(N_T)], axis=0)

a_rec_all = np.concatenate([a_rec_known, a_rec_unknown], axis=1)

#Multiple points reconstructed values
a_rec_known = np.concatenate([np.concatenate([np.concatenate([np.stack([(pathfinder.U_raw[:,t,m,:,:].transpose(0,2,1) @ (pathfinder.v_ARD[:, np.newaxis,:] * np.eye(D)[np.newaxis,:,:]) @ pathfinder.V_raw[:,t,m,:,:])[:, Idx_known[:,0], Idx_known[:,1]] for m in range(M-1)], axis=1), 
                ((pathfinder.U_raw[:,t,-1,:,:].transpose(0,2,1) @ (pathfinder.v_ARD[:, np.newaxis,:] * np.eye(D)[np.newaxis,:,:]) @ pathfinder.U_raw[:,t,-1,:,:])[:,Idx_known[:,0], Idx_known[:,1]][:,np.newaxis,:])], axis=1), 
                np.stack([(pathfinder.U_raw[:,t,m,:,:].transpose(0,2,1) @ (pathfinder.v_ARD[:, np.newaxis,:] * np.eye(D)[np.newaxis,:,:]) @ pathfinder.V_raw[:,t,m,:,:])[:,Idx_known[:,1], Idx_known[:,0]] for m in range(M-1)], axis=1 )[:,::-1,:]], axis=1) for t in range(N_T)], axis=1)

a_rec_unknown = np.concatenate([np.concatenate([np.concatenate([np.stack([(pathfinder.U_raw[:,t,m,:,:].transpose(0,2,1) @ (pathfinder.v_ARD[:, np.newaxis,:] * np.eye(D)[np.newaxis,:,:]) @ pathfinder.V_raw[:,t,m,:,:])[:, Idx_unknown[:,0], Idx_unknown[:,1]] for m in range(M-1)], axis=1), 
                ((pathfinder.U_raw[:,t,-1,:,:].transpose(0,2,1) @ (pathfinder.v_ARD[:, np.newaxis,:] * np.eye(D)[np.newaxis,:,:]) @ pathfinder.U_raw[:,t,-1,:,:])[:,Idx_unknown[:,0], Idx_unknown[:,1]][:,np.newaxis,:])], axis=1), 
                np.stack([(pathfinder.U_raw[:,t,m,:,:].transpose(0,2,1) @ (pathfinder.v_ARD[:, np.newaxis,:] * np.eye(D)[np.newaxis,:,:]) @ pathfinder.V_raw[:,t,m,:,:])[:,Idx_unknown[:,1], Idx_unknown[:,0]] for m in range(M-1)], axis=1 )[:,::-1,:]], axis=1) for t in range(N_T)], axis=1)

a_rec_all = np.concatenate([a_rec_known, a_rec_unknown], axis=2)