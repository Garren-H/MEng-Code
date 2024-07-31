
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
            matrix[N, 4] TT1 = append_col(append_col(append_col(rep_vector(1.0, N), T1), T1.^2), 1e-3 * T1.^3);
            matrix[M, 4] TT2 = append_col(append_col(append_col(rep_vector(1.0, M), T2), T2.^2), 1e-3 * T2.^3);

            return TT1 * TT2';
        }

        // Combined kernel
        matrix Kernel(vector x1, vector x2, vector T1, vector T2, int order) {
            return Kx(x1, x2, order) .* KT(T1, T2); 
        }
        // functions for the priors of the feature matrices
        real ps_feature_matrices(array[] int M_slice, int start, int end, array[] matrix U_raw, 
          array[] matrix V_raw) {
            real all_target = 0;
            for (m in start:end) {
                all_target += std_normal_lpdf(to_vector(U_raw[m]));
                all_target += std_normal_lpdf(to_vector(V_raw[m]));
            }
            return all_target;
          }
        // function for the likelihood and prior on smoothed values
        real ps_like(array[] int N_slice, int start, int end, vector y, matrix cov_y_yMC,
          matrix mu_y_y_MC, matrix L_y_MC_inv_cov, matrix y_MC_prec, vector v,
          array[,] int Idx_all, int M, int N_T, int N_MC, int N_C, int N_known, array[] int N_points,
          vector v_ARD, array[,] matrix U_raw, array[,] matrix V_raw, real v_MC, matrix K_MC) {
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
                all_target += -0.5 * dot_self(L_y_MC_inv_cov * y_MC_pred); // GP prior

                if (i <= N_known) {
                    vector[N_points[i]] y_pred = ( mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * y_MC_prec) * y_MC_pred;
                    matrix[N_points[i], N_points[i]] stable_inv; 
                    {
                        matrix[N_points[i], N_MC] ss = mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * L_y_MC_inv_cov';
                        stable_inv = ss * ss';
                        stable_inv = (stable_inv + stable_inv')/2.0; // insure symmetry
                    }
                    matrix[N_points[i], N_points[i]] cov_y = cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] - stable_inv;
                    cov_y = add_diag(cov_y, v[i]);
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
        vector[N_C] x2_int;                 // unique compositions to interpolate
        real<lower=0> scale_upper;          // upper bound for scale parameter
        int grainsize;                      // number of grainsizes

        int N;                              // number of components
        int D;                              // rank of feature matrices
        array[N_known, 2] int Idx_known;    // indices (row, column) of known datasets
        array[N_unknown, 2] int Idx_unknown;// indices (row, column) of unknown datasets
        
        real<lower=0> jitter;               // jitter for stability of covariances
    }

    transformed data {
        real error = 0.01;                                      // error in the data (fraction of experimental data)
        vector[sum(N_points)] var_data = square(error*y1);      // variance of the data
        int M = (N_C + 1) %/% 2;                                // interger division to get the number of U matrices
        int N_MC = N_C*N_T;                                     // overall number of interpolated datapoints per dataset
        array[M-1] int M_slice;                                 // array of integers to be used as indices in parallel computations
        array[N_known+N_unknown] int N_slice;                   // array of integers to be used as indices in parallel computations
        array[N_known+N_unknown,2] int Idx_all = append_array(Idx_known, Idx_unknown); // indices of all datasets
        vector[N_MC] x2;                                        // concatnated vector of x2_int
        vector[N_MC] T2;                                        // concatenated vector of T2_int
        matrix[N_MC, N_MC] K_MC;                                // kernel for the interpolated data
        matrix[sum(N_points), N_MC] mu_y_y_MC;                  // Mapping from y_MC to y (predictive GP mean)
        matrix[sum(N_points), max(N_points)] cov_y_yMC;         // covariance of y smmooth|y_MC
        matrix[N_MC, N_MC] cov_y_MC;                            // conditional covariance of y_MC|(U,V)
        // Assign MC input vectors for temperature and composition
        for (i in 1:N_T) {
            x2[(i-1)*N_C+1:i*N_C] = x2_int;
            T2[(i-1)*N_C+1:i*N_C] = rep_vector(T2_int[i],N_C);
        } 

        // Assign MC kernel
        K_MC = add_diag(Kernel(x2, x2, T2, T2, order), jitter);

        {
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
                // obtain kernel for the data
                matrix[N_points[i], N_points[i]] K_y = Kernel(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                            x1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                            T1[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                            T1[sum(N_points[:i-1])+1:sum(N_points[:i])], order);
                // Mapping from y_MC to y
                mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]), :] = K_y_yMC ;
                // Computation of K12*K22^-1*K12^T
                cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] = add_diag(K_y, var_data[sum(N_points[:i-1])+1:sum(N_points[:i])]);    
            }
        }

        for ( i in 1:N_known+N_unknown) {        
            // slice variable for parallel computations
            N_slice[i] = i;
        }

        // Slice variables for feature matrices
        M_slice = N_slice[:M-1];
    }

    parameters {
        vector<lower=0, upper=5>[N_known] v;        // data-model variance; constrained
        real<lower=0, upper=5> v_MC;                // error between of y_MC and (U,V); constrained
        array[N_T, M] matrix[D,N] U_raw;            // feature matrices U
        array[N_T, M-1] matrix[D,N] V_raw;          // feature matrices V
        vector<lower=0>[D] sigma_ARD;               // ARD variances on decorrelated prior
    }

    transformed parameters {
        vector[D] v_ARD = (scale_upper * sigma_ARD) .^ 2; // effective ARD variances; sqrt(v_ARD) ~ half-cauchy(0, scale_upper)
    }
    
    model {
        // Inverse cholesky factor of MC covariance
        matrix[N_MC, N_MC] L_y_MC_inv_cov = inverse(cholesky_decompose(add_diag(K_MC, v_MC)));
        // Precision matrix of MC
        matrix[N_MC, N_MC] y_MC_prec = crossprod(L_y_MC_inv_cov);         // L_MC_inv' * L_MC_inv

        // add addjustment for v_MC
        target += (N_known + N_unknown) * log_determinant(L_y_MC_inv_cov);
             
        // half-cauhcy prior for standard deviation of the feature matrices
        sigma_ARD ~ cauchy(0, 1);
    
        // Exponential prior for variance-model mismatch
        v ~ exponential(1);
        
        // inverse gamma prior for the error between the reconstructions and smoothed values
        v_MC ~ exponential(1);
        
        // priors for feature matrices
        for (t in 1:N_T) {
            target += reduce_sum(ps_feature_matrices, M_slice, grainsize, U_raw[t,:M-1], 
                        V_raw[t,:]);
            to_vector(U_raw[t,M,:,:]) ~ std_normal();
        }
        
        // Likelihood function
        target += reduce_sum(ps_like, N_slice, grainsize, y1, cov_y_yMC,
                        mu_y_y_MC, L_y_MC_inv_cov, y_MC_prec, v,
                        Idx_all, M, N_T, N_MC, N_C, N_known, N_points,
                        v_ARD, U_raw, V_raw, v_MC, K_MC);
    }

    generated quantities {
        vector[sum(N_points)] y_pred;
        vector[N_known] log_lik;
        matrix[N_MC, N_known+N_unknown] y_MC_pred;
        for (i in 1:N_known) {
            // Inverse cholesky factor of MC covariance
            matrix[N_MC, N_MC] L_y_MC_inv_cov = inverse(cholesky_decompose(add_diag(K_MC, v_MC)));
            // Precision matrix of MC
            matrix[N_MC, N_MC] y_MC_prec = crossprod(L_y_MC_inv_cov);         // L_MC_inv' * L_MC_inv
            // Stable computation of covariance
            matrix[N_points[i], N_MC] stable_inv = mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * L_y_MC_inv_cov'; 
            // Compute K2 - K21*K1^-1*K21^T
            matrix[N_points[i], N_points[i]] cov_y = cov_y_yMC[sum(N_points[:i-1])+1:sum(N_points[:i]), :N_points[i]] - stable_inv * stable_inv';
            // Add variance on diagonal cov_y = K2 - K21*K1^-1*K21^T + v2*I
            cov_y = add_diag(cov_y, v[i]);
            for(t in 1:N_T) {
                for (m in 1:M-1) {
                    y_MC_pred[m+N_C*(t-1), i] = dot_product(U_raw[t,m,:,Idx_all[i,1]] .* v_ARD, V_raw[t,m,:,Idx_all[i,2]]);
                    y_MC_pred[N_C-m+1+N_C*(t-1), i] = dot_product(U_raw[t,m,:,Idx_all[i,2]] .* v_ARD, V_raw[t,m,:,Idx_all[i,1]]); 
                }
                y_MC_pred[M+N_C*(t-1), i] = dot_product(U_raw[t,M,:,Idx_all[i,1]] .* v_ARD, U_raw[t,M,:,Idx_all[i,2]]);
            }
            y_pred[sum(N_points[:i-1])+1:sum(N_points[:i])] = ( mu_y_y_MC[sum(N_points[:i-1])+1:sum(N_points[:i]),:] * y_MC_prec) * y_MC_pred[:,i];
            log_lik[i] = multi_normal_lpdf(y1[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_pred[sum(N_points[:i-1])+1:sum(N_points[:i])], cov_y);
        }

        for (i in 1:N_unknown) {
            for (t in 1:N_T) {
                for (m in 1:M-1) {
                    y_MC_pred[m+N_C*(t-1), N_known+i] = dot_product(U_raw[t,m,:,Idx_all[N_known+i,1]] .* v_ARD, V_raw[t,m,:,Idx_all[N_known+i,2]]);
                    y_MC_pred[N_C-m+1+N_C*(t-1), N_known+i] = dot_product(U_raw[t,m,:,Idx_all[N_known+i,2]] .* v_ARD, V_raw[t,m,:,Idx_all[N_known+i,1]]); 
                }
                y_MC_pred[M+N_C*(t-1), N_known+i] = dot_product(U_raw[t,M,:,Idx_all[N_known+i,1]] .* v_ARD, U_raw[t,M,:,Idx_all[N_known+i,2]]);
            }
        }
    }
    