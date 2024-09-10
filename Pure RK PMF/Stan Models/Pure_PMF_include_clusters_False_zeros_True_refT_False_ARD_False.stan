
        functions {
            // convert array of matrix to array
            array [] real array_1D_mat_to_array(array[] matrix A) {
                array[3] int s = dims(A);
                array[s[1], s[2], s[3]] real A_array;
                for (i in 1:s[1]) {
                    A_array[i] = to_array_2d(A[i]);
                }
                return to_array_1d(A_array);
            }
            array [] real array_2D_mat_to_array(array[,] matrix A) {
                array[4] int s = dims(A);
                array[s[1], s[2], s[3], s[4]] real A_array;
                for (i in 1:s[1]) {
                    for (j in 1:s[2]) {
                        A_array[i,j] = to_array_2d(A[i,j]);
                    }
                }
                return to_array_1d(A_array);
            }
            
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
        }

        data {
            int N_known;                        // number of known mixtures
            int N_unknown;                      // number of unknown mixtures
            array[N_known] int N_points;        // number of experimental points per known dataset
            int order;                          // order of the compositional polynomial
            vector[sum(N_points)] x1;           // experimental composition
            vector[sum(N_points)] T1;           // experimental temperature 
            vector[sum(N_points)] y1;           // experimental excess enthalpy
            int N_T;                            // number of interpolated temperatures
            int N_C;                            // number of interpolated compositions
            vector[N_T] T2_int;                 // unique temperatures to interpolate
            vector[N_C] x2_int;                 // unique compositions to interpolate
            real<lower=0> scale_cauchy;         // scale for cauchy prior
            real<lower=0> v_MC;                 // error between of y_MC and (U,V)
            int grainsize;                      // number of grainsizes

            int N;                              // number of components
            int D;                              // rank of feature matrices
            array[N_known, 2] int Idx_known;    // indices (row, column) of known datasets
            array[N_unknown, 2] int Idx_unknown;// indices (row, column) of unknown datasets
            
            real<lower=0> jitter;               // jitter for stability of covariances
            vector<lower=0>[N_known] v;         // known data-model variance
            vector<lower=0>[D] v_ARD;           // ARD variances
        }

        transformed data {
            real error = 0.01;                  // error in the data (fraction of experimental data)
            int N_MC = N_C*N_T+2*N_T;  
            vector[N_MC] x2;                                                                // concatnated vector of x2_int
            vector[N_MC] T2;                                                                // concatenated vector of T2_int              
            vector[sum(N_points)] var_data = square(error*y1);                              // variance of the data
            int M = (N_C) %/% 2;                                                            // interger division to get the number of U matrices
            array[N_known+N_unknown] int N_slice;                                           // array of integers to be used as indices in parallel computations
            array[N_known+N_unknown,2] int Idx_all = append_array(Idx_known, Idx_unknown);  // indices of all datasets
            array[N_known] int N_points_ad;                                                 // adjusted number of datapoints (with interpolated compositions)
            matrix[sum(N_points)+N_known*N_MC, max(N_points)+N_MC] L_all_inv;               // inverse of the cholesky decomposition of the covariance matrix of all known mixtures
            matrix[N_MC, N_MC] L_MC_inv;                                                    // inverse of the cholesky decomposition of the covariance matrix of the interpolated compositions
            // Assign MC input vectors for temperature and composition
            for (i in 1:N_T) {
                x2[(i-1)*N_C+1:i*N_C] = x2_int;
                T2[(i-1)*N_C+1:i*N_C] = rep_vector(T2_int[i],N_C);
                x2[N_T*N_C+2*(i-1)+1] = 0;
                x2[N_T*N_C+2*(i-1)+2] = 1;
                T2[N_T*N_C+2*(i-1)+1] = T2_int[i];
                T2[N_T*N_C+2*(i-1)+2] = T2_int[i];
            } 

            L_MC_inv = inverse(cholesky_decompose(add_diag(Kernel(x2, x2, T2, T2, order), jitter+v_MC)));

            {
                matrix[sum(N_points)+N_known*(N_MC), max(N_points)+N_MC] cov_all;
                for (i in 1:N_known) {
                    N_points_ad[i] = N_points[i] + N_MC;
                    vector[N_points_ad[i]] x1_ad = append_row(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], x2);
                    vector[N_points_ad[i]] T1_ad = append_row(T1[sum(N_points[:i-1])+1:sum(N_points[:i])], T2);
                    cov_all[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]] = add_diag(Kernel(x1_ad, x1_ad, T1_ad, T1_ad, order), jitter);
                    {
                        vector[N_points_ad[i]] ad_err = append_row(var_data[sum(N_points[:i-1])+1:sum(N_points[:i])]+v[i], v_MC*rep_vector(1, N_MC));
                        cov_all[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]] += diag_matrix(ad_err);
                    }
                    L_all_inv[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]] = cholesky_decompose(cov_all[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]]);
                    L_all_inv[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]] = inverse(L_all_inv[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]]);
                }
            }
        }

        parameters {
            array[N_T,M] matrix[D,N] U_raw;         // feature matrices U
            array[N_T,M] matrix[D,N] V_raw;         // feature matrices V
        }
        
        model {
            array[N_T, M] matrix[D, N] U;
            array[N_T, M] matrix[D, N] V;    
            for (t in 1:N_T) {
                for (m in 1:M) {
                    U[t,m] = U_raw[t,m];
                    V[t,m] = V_raw[t,m]; 
                    V[t,m] = diag_pre_multiply(v_ARD, V[t,m]);     
                }
            }

            // half-cauhcy prior for standard deviation of the feature matrices
            v_ARD ~ cauchy(0, scale_cauchy); 
            target += sum(log(v_ARD[2:]-v_ARD[:D-1]+1e-30))+log(v_ARD[1]+1e-30);// Jacobian adjustement    

            // priors for feature matrices
            array_2D_mat_to_array(U_raw) ~ std_normal();
            array_2D_mat_to_array(V_raw) ~ std_normal();

            // Likelihood function
            {
                real all_target = 0;
                for (i in 1:N_known) {
                    vector[N_MC] y_MC_pred;
                    for (t in 1:N_T) {
                        real y_0 = 0;
                        real y_1 = 0;
                        for (m in 1:M) {
                            y_MC_pred[m+N_C*(t-1)] = dot_product(U[t,m,:,Idx_known[i,1]], V[t,m,:,Idx_known[i,2]]);
                            y_MC_pred[N_C-m+1+N_C*(t-1)] = dot_product(U[t,m,:,Idx_known[i,2]], V[t,m,:,Idx_known[i,1]]);
                            y_0 += dot_product(U[t,m,:,Idx_known[i,1]], V[t,m,:,Idx_known[i,1]])^2;
                            y_1 += dot_product(U[t,m,:,Idx_known[i,2]], V[t,m,:,Idx_known[i,2]])^2;
                        }
                        y_MC_pred[N_C*N_T+2*(t-1)+1] = y_0;
                        y_MC_pred[N_C*N_T+2*(t-1)+2] = y_1;
                    }
                    vector[N_points[i]+N_MC] y_all;
                    y_all[:N_points[i]] = y1[sum(N_points[:i-1])+1:sum(N_points[:i])]; 
                    y_all[N_points[i]+1:] = y_MC_pred;
                    all_target += -0.5 * dot_self(L_all_inv[sum(N_points_ad[:i-1])+1:sum(N_points_ad[:i]), :N_points_ad[i]] * y_all);
                }

                for (i in 1:N_unknown) {
                    vector[N_MC] y_MC_pred;
                    for (t in 1:N_T) {
                        real y_0 = 0;
                        real y_1 = 0;
                        for (m in 1:M) {
                            y_MC_pred[m+N_C*(t-1)] = dot_product(U[t,m,:,Idx_unknown[i,1]], V[t,m,:,Idx_unknown[i,2]]);
                            y_MC_pred[N_C-m+1+N_C*(t-1)] = dot_product(U[t,m,:,Idx_unknown[i,2]], V[t,m,:,Idx_unknown[i,1]]);
                            y_0 += dot_product(U[t,m,:,Idx_unknown[i,1]], V[t,m,:,Idx_unknown[i,1]])^2;
                            y_1 += dot_product(U[t,m,:,Idx_unknown[i,2]], V[t,m,:,Idx_unknown[i,2]])^2;
                        }
                        y_MC_pred[N_C*N_T+2*(t-1)+1] = y_0;
                        y_MC_pred[N_C*N_T+2*(t-1)+2] = y_1;
                    }
                    all_target += -0.5 * dot_self(L_MC_inv * y_MC_pred); // GP prior
                }
                target += all_target;
            } 
        }

        generated quantities {
            matrix[N_C*N_T, N_known+N_unknown] y_MC_pred;
            {
                array[N_T, M] matrix[D, N] U;
                array[N_T, M] matrix[D, N] V;  

                for (t in 1:N_T) {
                    for (m in 1:M) {
                        U[t,m] = U_raw[t,m];
                        V[t,m] = V_raw[t,m];
                        V[t,m] = diag_pre_multiply(v_ARD, V[t,m]);
                        for (i in 1:N_known+N_unknown) {
                            y_MC_pred[m+N_C*(t-1), i] = dot_product(U[t,m,:,Idx_all[i,1]], V[t,m,:,Idx_all[i,2]]);
                            y_MC_pred[N_C-m+1+N_C*(t-1), i] = dot_product(U[t,m,:,Idx_all[i,2]], V[t,m,:,Idx_all[i,1]]); 
                        }       
                    }
                }
            }
        }
        
    