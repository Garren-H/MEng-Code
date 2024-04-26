'''
Function file to create a valid stan model such 
that the variances are in increasing order for
the Hybrid model
'''

def generate_stan_code(D, include_clusters=False, variance_known=False):
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
        matrix K(vector x1, vector x2, vector T1, vector T2, int order) {
            return Kx(x1, x2, order) .* KT(T1, T2); 
        }'''
    if include_clusters:
        model_code += '''
        // functions for the priors of the feature matrices and means thereof
        real ps_feature_matrices(array[] M_slice, int start, int end, array[] matrix U_raw_means,
          array[] matrix V_raw_means, array[] U_raw, array[] V_raw, matrix E_cluster, matrix C) {
            real all_target;
            for (m in start:end) {
                all_target += std_normal(to_vector(U_raw_means[m]));
                all_target += std_normal(to_vector(V_raw_means[m]));
                all_target += normal_lupdf(to_vector(U_raw[m]) | to_vector(U_raw_means[m]*C), to_vector(E_cluster));
                all_target += normal_lupdf(to_vector(V_raw[m]) | to_vector(U_raw_means[m]*C), to_vector(E_cluster));
            }
            return all_target;
          }
        '''
    else:
        model_code += '''
        // functions for the priors of the feature matrices
        real ps_feature_matrices(array[] M_slice, int start, int end, array[] U_raw, 
          array[] V_raw) {
            real all_target;
            for (m in start:end) {
                all_target += std_normal(to_vector(U_raw[m]));
                all_target += std_normal(to_vector(V_raw[m]));
            }
            return all_target;
          }
        '''
    
    if variance_known: 
        model_code += '''
        // function for the likelihood and prior on smoothed values
        real ps_like(array[] N_slice, int start, int end, vector y, matrix cov_y_yMC, matrix cov_y_MC,
          array[,] Idx_all, int M, int N_T, int N_MC, int N_C, matrix K_MC_inv, int N_known, array[] N_points,
          vector v_ARD, array[,] matrix U_raw, array[,] matrix V_raw, matrix y_MC, matrix A_y_yMC) {
            real all_target;
            for (i in start:end) {
                vector[N_MC] y_MC_pred;
                for(t in 1:N_T) {
                    for (m in 1:M-1) {
                        y_MC_pred[m+N_C*(t-1)] = dot_product(U_raw[t,m,:,Idx_all[i,1]] .* v_ARD, V_raw[t,m,:,Idx_all[i,2]]);
                        y_MC_pred[M-m+1+N_C*(t-1)] = dot_product(U_raw[t,m,:,Idx_all[i,2]] .* v_ARD, V_raw[t,m,:,Idx_all[i,1]]); 
                    }
                    y_MC_pred[M+N_C*(t-1)] = dot_product(U_raw[t,M,:,Idx_all[i,1]] .* v_ARD, U_raw[t,M,:,Idx_all[i,2]]);
                }
                all_target += multi_normal_lupdf(y_MC[:,i] | y_MC_pred, cov_y_MC);

                if (i <= N_known) {
                    vector[N_points[i]] y_pred = A_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * y_MC[:,i];
                    all_target += multi_normal_lupdf(y[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_pred, cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]]);
                }
            }
            return all_target;
          }'''
    else:
        model_code += '''
        // function for the likelihood and prior on smoothed values
        real ps_like(array[] N_slice, int start, int end, vector y, matrix cov_y_yMC, matrix cov_y_MC,
          array[,] Idx_all, int M, int N_T, int N_MC, int N_C, matrix K_MC_inv, int N_known, array[] N_points,
          vector v_ARD, array[,] matrix U_raw, array[,] matrix V_raw, matrix y_MC, matrix A_y_yMC) {
            real all_target;
            for (i in start:end) {
                vector[N_MC] y_MC_pred;
                for(t in 1:N_T) {
                    for (m in 1:M-1) {
                        y_MC_pred[m+N_C*(t-1)] = dot_product(U_raw[t,m,:,Idx_all[i,1]] .* v_ARD, V_raw[t,m,:,Idx_all[i,2]]);
                        y_MC_pred[M-m+1+N_C*(t-1)] = dot_product(U_raw[t,m,:,Idx_all[i,2]] .* v_ARD, V_raw[t,m,:,Idx_all[i,1]]); 
                    }
                    y_MC_pred[M+N_C*(t-1)] = dot_product(U_raw[t,M,:,Idx_all[i,1]] .* v_ARD, U_raw[t,M,:,Idx_all[i,2]]);
                }
                all_target += multi_normal_lupdf(y_MC[:,i] | y_MC_pred, cov_y_MC);

                if (i <= N_known) {
                    vector[N_points[i]] y_pred = A_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * y_MC[:,i];
                    all_target += multi_normal_lupdf(y[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_pred, add_diag(cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]],v[i]));
                }
            }
            return all_target;
          }'''

    model_code += '''
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
        vector[N_C] x2_int;                 // unique compositions to interpolate
        real<lower=0> v_MC;                 // error between of y_MC and (U,V)
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
        vector[sum(N_points)] var_data = square(0.01*y);                    // variance of the data
        int M = (N_C + 1) %/% 2;                                // interger division to get the number of U matrices
        int N_MC = N_C*N_T;                                     // overall number of interpolated datapoints per dataset
        array[M-1] int M_slice;                                 // array of integers to be used as indices in parallel computations
        array[N_known+N_unknown] int N_slice;                   // array of integers to be used as indices in parallel computations
        array[N_known+N_unknown] int Idx_all = append_row(Idx_known, Idx_unknown); // indices of all datasets
        vector[N_MC] x2;                                        // concatnated vector of x2_int
        vector[N_MC] T2;                                        // concatenated vector of T2_int
        matrix[N_MC, N_MC] K_MC;                                // kernel for the interpolated data
        matrix[N_MC, N_MC] K_MC_inv;                            // inverse of kernel matrix
        matrix[sum(N_points), max(N_points)] K_y;               // kernel for the experimental data
        matrix[sum(N_points), N_MC] K_y_yMC;                    // kernel between experimental and interpolated data
        matrix[sum(N_points), N_MC] A_y_yMC;                    // Mapping from y_MC to y (predictive GP mean)
    '''
    
    if include_clusters: # transform cluster variance parameters into a vector for easier processing
        model_code += '''
        matrix[D, N] E_cluster = rep_matrix(v_cluster', D) * C; // within cluster varaince matrix'''
    
    if variance_known: # include known data-model variance data input
         model_code += '''
         matrix[sum(N_points), max(N_points)] cov_y;            // covariance matrix for the data
         matrix[sum(N_points), max(N_points)] cov_y_yMC;        // conditional covariance matrix y|y_MC''' 

    model_code += '''
        matrix[N_MC, N_MC] cov_y_MC;                            // conditional covariance of y_MC|(U,V)
        // Assign MC input vectors for temperature and composition
        for (i in 1:N_T) {
            x2[(i-1)*N_C+1:i*N_C] = x2_int;
            T2[(i-1)*N_C+1:i*N_C] = rep_vector(T2_int[i],N_C);
        } 

        // Assign MC kernel
        K_MC = add_diag(K(x2, x2, T2, T2, order), jitter);
        K_MC_inv = inverse_spd(K_MC);


        // stable version to compute the covariance cov_y_MC
        {
            matrix[N_MC, N_MC] L_MC = cholesky_decompose(add_diag(K_MC, v_MC));
            matrix[N_MC, N_MC] L_MC_inv = inverse(L_MC);
            matrix[N_MC, N_MC] stable_inv = K_MC * L_MC_inv'
            cov_y_MC = K_MC - stable_inv * stable_inv';
        }

        // Assign experimental data kernel and kernel between experimental and interpolated data
        for (i in 1:N_known) {
            K_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = add_diag(K(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], x1[sum(N_points[:i-1])+1:sum(N_points[:i])], T1[sum(N_points[:i-1])+1:sum(N_points[:i])], T1[sum(N_points[:i-1])+1:sum(N_points[:i])], order), jitter);
            K_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] = K(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], x2, T1[sum(N_points[:i-1])+1:sum(N_points[:i])], T2, order);
            A_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] = K_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] * K_MC_inv;'''
    if variance_known:
        model_code += '''
            cov_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = add_diag(K_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]], var_data[sum(N_points[:i-1])+1:sum(N_points[:i])]+v[i]);
            
            // stable version of the computation of cov_y_MC
            {
                matrix[N_MC, N_MC] L_MC = cholesky_decompose(K_MC);
                matrix[N_MC, N_MC] L_MC_inv = inverse(L_MC);
                matrix[N_points[i], N_MC] stable_inv =  K_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] * L_MC_inv';
                cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = cov_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] - stable_inv * stable_inv';
            }'''
    else:
        model_code += '''
            cov_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = K_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]], var_data[sum(N_points[:i-1])+1:sum(N_points[:i])];
            
            // stable version of the computation of cov_y_MC
            {
                matrix[N_MC, N_MC] L_MC = cholesky_decompose(K_MC);
                matrix[N_MC, N_MC] L_MC_inv = inverse(L_MC);
                matrix[N_points[i], N_MC] stable_inv =  K_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] * L_MC_inv';
                cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = cov_y[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] - stable_inv * stable_inv';
            }'''
    model_code += '''

            N_slice[i] = i;
        }

        for (i in 1:N_unknown) {
            N_slice[N_known+i] = N_known+i;
        }

        M_slice = N_slice[:M-1];
    }

    parameters {'''

    if not variance_known: # include data-model variance as parameter to model
        model_code += '''
        vector<lower=0>[N_known] v;                 // data-model variance'''

    if include_clusters: # include cluster means as parameters
        model_code += '''
        array[N_T, M] matrix[D,K] U_raw_means;      // U_raw cluster means
        array[N_T, M-1] matrix[D,K] V_raw_means;    // V_raw cluster means'''

    model_code += '''
        array[N_T, M] matrix[D,N] U_raw;            // feature matrices U
        array[N_T, M-1] matrix[D,N] V_raw;          // feature matrices V
        real<lower=0, upper=5> scale;               // scale dictating the strenght of ARD effect'''

    # generate different parameters for each ARD variance parameter, and lower bound by previous
    model_code += '''
        real<lower=0> v_ARD_1;                      // ARD variance parameter 1'''
    for d in range(1,D):
        model_code += f'''
        real<lower=v_ARD_{d}> v_ARD_{d+1};          // ARD variance parameter {d+1}'''
    model_code += '''

        matrix[N_MC, N_known+N_unknown] y_MC;       // smoothed interpolated values
    }
    '''

    # create transformed parameters block where the ARD varaicnes are combined
    model_code += '''
    transformed parameters {'''
    model_code += f'''
        vector<lower=0>[D] v_ARD = ['''
    for d in range(1,D):
        model_code += f'v_ARD_{d}, '
    model_code += f"v_ARD_{D}]'; // ARD variance vector"
    model_code += '''
    }
    '''

    model_code += '''
    model {
        // exponential prior
        v_ARD ~ exponential(scale);
    '''

    if not variance_known: # include data-model variance as parameter prior to model
        model_code += '''
        // exponential prior for variance-model mismatch
        v ~ exponential(2);
        '''

    if include_clusters: # include cluster mean priors
        model_code += '''
        // priors for cluster means and feature matrices 
        for (t in 1:N_T) {
            target += reduce_sum(ps_feature_matrices, M_slice, grainsize, U_raw_means,
                        V_raw_means, U_raw, V_raw, E_cluster, C);
            to_vector(U_raw_means[t,M,:,:]) ~ std_normal();
            to_vector(U_raw[t,M,:,:]) ~ normal(to_vector(U_raw_means[t,M,:,:]*C), to_vector(E_cluster));
        }
        '''
    else: # exclude cluster parameters 
        model_code += '''
        // priors for feature matrices
        for (t in 1:N_T) {
            target += reduce_sum(ps_feature_matrices, M_slice, grainsize, U_raw, 
                        V_raw);
            to_vector(U_raw[t,M,:,:]) ~ std_normal();
        }
        '''

    model_code += '''
        // Likelihood function
        target += reduce_sum(ps_like, N_slice, grainsize, y, cov_y_yMC, cov_y_MC,
          Idx_all, M, N_T, N_MC, N_C, K_MC_inv, N_known, N_points,
          v_ARD, U_raw, V_raw, y_MC, A_y_yMC);
    }
    '''

    return model_code