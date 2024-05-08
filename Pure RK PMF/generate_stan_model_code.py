'''
Function file to create a valid stan model such 
that the variances are in increasing order for
the Hybrid model
'''

def generate_stan_code(D, include_clusters=False, variance_known=False, variance_MC_known=False):
    # D: number of features/lower rank of feature matrices
    # include_clusters: whether to include cluster data
    #                 : If true include number of cluster, cluster 
    #                   assignment as a matrix, and the within cluster varaince 
    #                   (vector for each cluster) as inputs to the model
    # variance_known: whether the data-model variance is known
    #               : If true include the data-model variance as input to the model

    model_code = '''
    functions {
        // kernel for composition
        matrix Kx(vector x1, vector x2, int order) {
            int N = rows(x1);
            int M = rows(x2);
            matrix[N, order+1] X1;
            matrix[M, order+1] X2;
            for (i in 1:order) {
                X1[:,i] = x1 .^(order+2-i) - x1;
                X2[:,i] = x2 .^(order+2-i) - x2;
            }
            X1[:,order+1] = 1e-1 * x1 .* sqrt(1-x1) .* exp(x1);
            X2[:,order+1] = 1e-1 * x2 .* sqrt(1-x2) .* exp(x2);
            return X1 * X2';
        }

        // kernel for Temperature
        matrix KT(vector T1, vector T2) {
            int N = rows(T1);
            int M = rows(T2);
            matrix[N, 4] TT1 = append_col(append_col(append_col(rep_vector(1.0, N), T1), 1e-2* T1.^2), 1e-4* T1.^3);
            matrix[M, 4] TT2 = append_col(append_col(append_col(rep_vector(1.0, M), T2), 1e-2* T2.^2), 1e-4* T2.^3);

            return TT1 * TT2';
        }

        // Combined kernel
        matrix Kernel(vector x1, vector x2, vector T1, vector T2, int order) {
            return Kx(x1, x2, order) .* KT(T1, T2); 
        }'''
    if include_clusters:
        model_code += '''
        // functions for the priors of the feature matrices and means thereof
        real ps_feature_matrices(array[] int M_slice, int start, int end, array[] matrix U_raw_means,
          array[] matrix V_raw_means, array[] matrix U_raw, array[] matrix V_raw, vector E_cluster, matrix C) {
            real all_target = 0;
            for (m in start:end) {
                all_target += std_normal_lpdf(to_vector(U_raw_means[m]));
                all_target += std_normal_lpdf(to_vector(V_raw_means[m]));
                all_target += normal_lpdf(to_vector(U_raw[m]) | to_vector(U_raw_means[m]*C), E_cluster);
                all_target += normal_lpdf(to_vector(V_raw[m]) | to_vector(U_raw_means[m]*C), E_cluster);
            }
            return all_target;
          }
        '''
    else:
        model_code += '''
        // functions for the priors of the feature matrices
        real ps_feature_matrices(array[] int M_slice, int start, int end, array[] matrix U_raw, 
          array[] matrix V_raw) {
            real all_target = 0;
            for (m in start:end) {
                all_target += std_normal_lpdf(to_vector(U_raw[m]));
                all_target += std_normal_lpdf(to_vector(V_raw[m]));
            }
            return all_target;
          }'''
    model_code += '''
        // function for the likelihood and prior on smoothed values
        real ps_like(array[] int N_slice, int start, int end, vector y, matrix cov_f2_f1,
          matrix mu_y_y_MC, matrix y_MC, matrix mu_y_MC, matrix prec_y_MC, vector var_data, vector v,
          array[,] int Idx_all, int M, int N_T, int N_MC, int N_C, int N_known, array[] int N_points,
          vector v_ARD, array[,] matrix U_raw, array[,] matrix V_raw) {
            real all_target = 0;
            for (i in start:end) {
                vector[N_MC] y_MC_pred;
                for(t in 1:N_T) {
                    for (m in 1:M-1) {
                        y_MC_pred[m+N_C*(t-1)] = dot_product(U_raw[t,m,:,Idx_all[i,1]] .* v_ARD, V_raw[t,m,:,Idx_all[i,2]]);
                        y_MC_pred[N_C-m+1+N_C*(t-1)] = dot_product(U_raw[t,m,:,Idx_all[i,2]] .* v_ARD, V_raw[t,m,:,Idx_all[i,1]]); 
                    }
                    y_MC_pred[M+N_C*(t-1)] = dot_product(U_raw[t,M,:,Idx_all[i,1]] .* v_ARD, U_raw[t,M,:,Idx_all[i,2]]);
                }
                all_target += multi_normal_prec_lpdf(y_MC[:,i] | mu_y_MC * y_MC_pred, prec_y_MC);

                if (i <= N_known) {
                    vector[N_points[i]] y_pred = mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * y_MC[:,i];
                    matrix[N_points[i], N_points[i]] cov_y = add_diag(cov_f2_f1[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]], var_data[sum(N_points[:i-1])+1:sum(N_points[:i])]+v[i]);
                    all_target += multi_normal_lpdf(y[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_pred, cov_y);
                }
            }
            return all_target;
          }
    }

    data {
        int N_known;                        // number of known mixtures
        int N_unknown;                      // number of unknown mixtures
        array[N_known] int N_points;        // number of experimental points per known dataset
        int order;                          // order of the compositional polynomial
        vector[sum(N_points)] x1;           // experimental composition
        vector[sum(N_points)] T1;           // experimental temperatures
        vector[sum(N_points)] y1;           // experimental excess enthalpy
        int N_T;                            // number of interpolated temperatures
        int N_C;                            // number of interpolated compositions
        vector[N_T] T2_int;                 // unique temperatures to interpolate
        vector[N_C] x2_int;                 // unique compositions to interpolate'''
    if variance_MC_known:
        model_code += '''
        real<lower=0> v_MC;                 // error between of y_MC and (U,V)'''
    model_code += '''
        int grainsize;                      // number of grainsizes

        int N; // number of components
        int D; // rank of feature matrices
        array[N_known, 2] int Idx_known;    // indices (row, column) of known datasets
        array[N_unknown, 2] int Idx_unknown;// indices (row, column) of unknown datasets
        
        real<lower=0> jitter;               // jitter for stability of covariances'''
    if variance_known: # include known data-model variance data input
        model_code += '''
        vector<lower=0>[N_known] v;         // known data-model variance'''

    if include_clusters: # include cluster data input
        model_code += '''   
        int K;                              // number of clusters
        matrix[K, N] C;                     // cluster assignment
        vector<lower=0>[K] v_cluster;       // within cluster variance'''

    model_code += '''
    }

    transformed data {
        real error = 0.01;                                      // error in the data (fraction of experimental data)
        vector[sum(N_points)] var_data = square(error*y1);       // variance of the data
        int M = (N_C + 1) %/% 2;                                // interger division to get the number of U matrices
        int N_MC = N_C*N_T;                                     // overall number of interpolated datapoints per dataset
        array[M-1] int M_slice;                                 // array of integers to be used as indices in parallel computations
        array[N_known+N_unknown] int N_slice;                   // array of integers to be used as indices in parallel computations
        array[N_known+N_unknown,2] int Idx_all = append_array(Idx_known, Idx_unknown); // indices of all datasets
        vector[N_MC] x2;                                        // concatnated vector of x2_int
        vector[N_MC] T2;                                        // concatenated vector of T2_int
        matrix[N_MC, N_MC] K_MC;                                // kernel for the interpolated data
        matrix[N_MC, N_MC] K_MC_inv;                            // inverse of kernel
        matrix[sum(N_points), N_MC] mu_y_y_MC;                  // Mapping from y_MC to y (predictive GP mean)
        matrix[sum(N_points), max(N_points)] cov_f2_f1;                  // covariance of y smmooth|y_MC'''
    if variance_MC_known:
        model_code+='''
        matrix[N_MC, N_MC] mu_y_MC;                             // Mapping from reconstructions to smooth MC
        matrix[N_MC, N_MC] prec_y_MC;                           // precision of y_MC'''
    
    if include_clusters: # transform cluster variance parameters into a vector for easier processing
        model_code += '''
        vector[D*N] E_cluster = to_vector((rep_matrix(v_cluster', D) * C)); // within cluster varaince matrix'''
    
    if variance_known: # include known data-model variance data input
        model_code += '''
        matrix[sum(N_points), max(N_points)] cov_y_yMC;        // conditional covariance matrix y|y_MC''' 

    model_code += '''
        matrix[N_MC, N_MC] cov_y_MC;                            // conditional covariance of y_MC|(U,V)
        // Assign MC input vectors for temperature and composition
        for (i in 1:N_T) {
            x2[(i-1)*N_C+1:i*N_C] = x2_int;
            T2[(i-1)*N_C+1:i*N_C] = rep_vector(T2_int[i],N_C);
        } 

        // Assign MC kernel
        K_MC = add_diag(Kernel(x2, x2, T2, T2, order), jitter);
        // Compute inverse using cholesky
        K_MC_inv = inverse_spd(K_MC);
        {
            matrix[N_MC, N_MC] L_MC = cholesky_decompose(K_MC); // Compute cholesky decomposition of K_MC
            matrix[N_MC, N_MC] L_MC_inv = inverse(L_MC);        // Inverse of cholesky decomposition
            // Loop through data sets to compute the relevant matrix, mapping from y_MC to y and covariance
            for (i in 1:N_known) {
                // obtain kernel matrix for cross covariance
                matrix[N_points[i], N_MC] K_y_yMC = Kernel(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                        x2, 
                                                        T1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                        T2, order); 

                // Stable implementation of A*B*A^T where B is a positive semi-definite matrix
                // This is decomposed into A*(L^-1)^T*L^-1*A^T where L is the cholesky factor of B
                // A*B*A^T = C*C^T where C = A*(L^-1)^T

                // Computation of above mentioned K12*K22^-1*K12^T
                matrix[N_points[i], N_MC] stable_inv = K_y_yMC * L_MC_inv';
                // obtain kernel for the data
                matrix[N_points[i], N_points[i]] K_y = Kernel(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                            x1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                            T1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                            T1[sum(N_points[:i-1])+1:sum(N_points[:i])], order);
                // Computation of K12*K22^-1*K12^T
                cov_f2_f1[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = K_y - stable_inv * stable_inv';
                // Mapping from y_MC to y
                mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] = K_y_yMC * K_MC_inv;

                // slice variable for parallel computations
                N_slice[i] = i;
            }'''
    if variance_MC_known:
        model_code += '''
            prec_y_MC = add_diag(K_MC_inv, v_MC^-1); // Precision matrix of y_MC
            prec_y_MC = (prec_y_MC+prec_y_MC')/2; // ensure symmetricness
            // Computation of the mapping of the reconstructed entries to the smoothed values
            {
               matrix[N_MC, N_MC] prec_y_MC_noisy = K_MC * inverse_spd(add_diag(K_MC, v_MC));
               mu_y_MC = prec_y_MC_noisy * add_diag(-prec_y_MC_noisy, 2);
            }'''
    model_code += '''
        }
        
        // Continue to assign the slice variable for unknown datasets
        for (i in 1:N_unknown) {
            N_slice[N_known+i] = N_known+i;
        }

        // Slice variables for feature matrices
        M_slice = N_slice[:M-1];
    }

    parameters {'''

    if not variance_known: # include data-model variance as parameter to model
        model_code += '''
        vector<lower=0>[N_known] v;                 // data-model variance'''

    if not variance_MC_known: # include MC variance as parameter to model
        model_code += '''
        real<lower=0> v_MC;                         // error between of y_MC and (U,V)'''

    if include_clusters: # include cluster means as parameters
        model_code += '''
        array[N_T, M] matrix[D,K] U_raw_means;      // U_raw cluster means
        array[N_T, M-1] matrix[D,K] V_raw_means;    // V_raw cluster means'''

    model_code += '''
        array[N_T, M] matrix[D,N] U_raw;            // feature matrices U
        array[N_T, M-1] matrix[D,N] V_raw;          // feature matrices V
        real<lower=0, upper=5> scale;               // scale dictating the strenght of ARD effect'''

    # ARD variance
    model_code += '''
        positive_ordered[D] v_ARD;        // ARD variances aranged in increasing order with lower bound zero
    }
    '''

    model_code += '''
    model {'''
    if not variance_MC_known:
        model_code += '''
        matrix[N_MC, N_MC] mu_y_MC;                                 // Mapping from reconstructions to smooth MC
        matrix[N_MC, N_MC] prec_y_MC = add_diag(K_MC_inv, v_MC^-1); // Precision matrix of y_MC
        prec_y_MC = (prec_y_MC+prec_y_MC')/2; // ensure symmetricness
        // Compute mapping from reconstructions to smooothed values
        {
            matrix[N_MC, N_MC] prec_y_MC_noisy = K_MC * inverse_spd(add_diag(K_MC, v_MC));
            mu_y_MC = prec_y_MC_noisy * add_diag(-prec_y_MC_noisy, 2);
        }'''
    model_code += '''
        // exponential prior on ARD variances
        v_ARD ~ exponential(scale);
    '''

    if not variance_known: # include data-model variance as parameter prior to model
        model_code += '''
        // exponential prior for variance-model mismatch
        v ~ exponential(2);
        '''

    if not variance_MC_known: # include MC variance as parameter prior to model
        model_code += '''
        // inverse gamma prior for the error between the reconstructions and smoothed values
        v_MC ~ inv_gamma(2,1);'''

    if include_clusters: # include cluster mean priors
        model_code += '''
        // priors for cluster means and feature matrices 
        for (t in 1:N_T) {
            target += reduce_sum(ps_feature_matrices, M_slice, grainsize, U_raw_means[t,:],
                        V_raw_means[t,:], U_raw[t,:M-1], V_raw[t,:M-1], E_cluster, C);
            to_vector(U_raw_means[t,M,:,:]) ~ std_normal();
            to_vector(U_raw[t,M,:,:]) ~ normal(to_vector(U_raw_means[t,M,:,:]*C), E_cluster);
        }
        '''
    else: # exclude cluster parameters 
        model_code += '''
        // priors for feature matrices
        for (t in 1:N_T) {
            target += reduce_sum(ps_feature_matrices, M_slice, grainsize, U_raw[t,:M-1], 
                        V_raw[t,:]);
            to_vector(U_raw[t,M,:,:]) ~ std_normal();
        }
        '''

    model_code += '''
        // Likelihood function
        target += reduce_sum(ps_like, N_slice, grainsize, y1, cov_f2_f1,
          mu_y_y_MC, y_MC, mu_y_MC, prec_y_MC, var_data, v,
          Idx_all, M, N_T, N_MC, N_C, N_known, N_points,
          v_ARD, U_raw, V_raw);
    }
    '''

    return model_code