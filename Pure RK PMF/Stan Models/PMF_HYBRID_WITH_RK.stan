
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
    }
}

data {
    int N_known; // number of known mixtures
    int N_unknown; // number of unknown mixtures
    array[N_known] int N_points; // number of experimental points per known dataset
    int order; // order of the compositional polynomial
    vector[sum(N_points)] x1; // experimental composition
    vector[sum(N_points)] T1; // experimental temperatures
    vector[sum(N_points)] y1; // experimental excess enthalpy
    int N_T; // number of interpolated temperatures
    int N_C; // number of interpolated compositions
    vector[N_T] T2_int; // unique temperatures to interpolate
    vector[N_C] x2_int; // unique compositions to interpolate
    real<lower=0, upper=1> alpha_lower; // lower bound on the contributions of priors
    real<lower=0, upper=1> alpha_upper; // upper bound on the contributions of priors

    int N; // number of components
    int D; // rank of feature matrices
    array[N_known, 2] int Idx_known; // indices (row, column) of known datasets
    array[N_unknown, 2] int Idx_unknown; // indices (row, column) of unknown datasets 
}

transformed data {
    real error=0.01;
    int M = (N_C + 1) %/% 2; // interger division to get the number of U matrices
    int N_MC = N_C*N_T; // overall number of interpolated datapoints per dataset
    vector[N_MC] x2; // concatnated vector of x2_int
    vector[N_MC] T2; // concatenated vector of T2_int
    matrix[N_MC, N_MC] K_MC; // kernel for the interpolated data
    matrix[sum(N_points), sum(N_points)] K_y; // kernel for the experimental data

    // Assign MC input vectors for temperature and composition
    for (i in 1:N_T) {
        x2[(i-1)*N_C+1:i*N_C] = x2_int;
        T2[(i-1)*N_C+1:i*N_C] = rep_vector(T2_int[i],N_C);
    } 

    // Assign MC kernel
    K_MC = K(x2, x2, T2, T2, order);

    // Assign experimental data kernel
    for (i in 1:N_known) {
        K_y[sum(N_points[:i-1])+1:sum(N_points[:i]), sum(N_points[:i-1])+1:sum(N_points[:i])] = K(x1[sum(N_points[:i-1])+1:sum(N_points[:i])], x1[sum(N_points[:i-1])+1:sum(N_points[:i])], T1[sum(N_points[:i-1])+1:sum(N_points[:i])], T1[sum(N_points[:i-1])+1:sum(N_points[:i])], order);
    }
}

parameters {
    // MC parameters
    array[N_T, M*2-1] matrix[D,N] F_raw; // scaled feature matrices; M U matrices, and M-1 V matrices
    vector<lower=0>[D] v_ARD; // variance of the ARD framework
    real<lower=0> scale; // scale parameter dictating strenght of ARD effect

    // Prior Contributon parameters
    real<lower=alpha_lower, upper=alpha_upper> alpha; // parameter for the contribution of the priors

    // Data parameters
    vector<lower=0>[N_known+N_unknown] v_D; // variance for the data-model mismatch.
}

model {
    matrix[N_known+N_unknown, N_MC] y_MC; // all the interpolated data for all the datasets 

    // prior on scale
    scale ~ gamma(2,1);

    // prior on v_ARD
    v_ARD ~ exponential(scale);

    // prior on alpha

    //Assignment of y_MC for known datasets
    for (n in 1:N_known) {
        int counter = 1;
        for (t in 1:N_T) {
            // composition up until just before 0.5
            for (m in 1:M-1) {
                y_MC[n, counter] = F_raw[t,m*2-1,:,Idx_known[n,1]]' * diag_matrix(v_ARD) * F_raw[t,m*2,:,Idx_known[n,2]];
                counter += 1;
            }
            // composition of 0.5
            y_MC[n, counter] = F_raw[t,2*M-1,:,Idx_known[n,1]]' * diag_matrix(v_ARD) * F_raw[t,2*M-1,:,Idx_known[n,2]];
            counter += 1;
            // composition from just after 0.5
            for (m in 1:M-1) {
                y_MC[n, counter] = F_raw[t,(M-m)*2-1,:,Idx_known[n,2]]' * diag_matrix(v_ARD) * F_raw[t,(M-m)*2,:,Idx_known[n,1]];
                counter += 1;
            }
        }
    }

    //Assignment of y_MC for unknown datasets
    for (n in 1:N_unknown) {
        int counter = 1;
        for (t in 1:N_T) {
            // composition up until just before 0.5
            for (m in 1:M-1) {
                y_MC[N_known+n, counter] = F_raw[t,m*2-1,:,Idx_unknown[n,1]]' * diag_matrix(v_ARD) * F_raw[t,m*2,:,Idx_unknown[n,2]];
                counter += 1;
            }
            // composition of 0.5
            y_MC[N_known+n, counter] = F_raw[t,2*M-1,:,Idx_unknown[n,1]]' * diag_matrix(v_ARD) * F_raw[t,2*M-1,:,Idx_unknown[n,2]];
            counter += 1;
            // composition from just after 0.5
            for (m in 1:M-1) {
                y_MC[N_known+n, counter] = F_raw[t,(M-m)*2-1,:,Idx_unknown[n,2]]' * diag_matrix(v_ARD) * F_raw[t,(M-m)*2,:,Idx_unknown[n,1]];
                counter += 1;
            }
        }
    }

    // Prior on the data-model mismatch
    v_D ~ exponential(1);

    // Priors for MC with constribution weighting
    for (t in 1:N_T) {
        for (m in 1:M-1) {
            target += alpha * std_normal_lpdf(to_vector(F_raw[t,2*m-1,:,:]));
            target += alpha * std_normal_lpdf(to_vector(F_raw[t,2*m,:,:]));
        }
        target += alpha * std_normal_lpdf(to_vector(F_raw[t,2*M-1,:,:]));
    }

    // Priors for GP with constribution
    for (n in 1:N_known+N_unknown) {
        matrix[N_MC, N_MC] cov_MC = K_MC + diag_matrix(rep_vector(v_D[n], N_MC));
        target += (1-alpha) * multi_normal_lpdf(y_MC[n,:]' | rep_vector(0.0, N_MC), cov_MC);
    } 

    // Likelihood
    for (n in 1:N_known) {
        matrix[N_MC, N_MC] cov_MC = K_MC + diag_matrix(rep_vector(v_D[n], N_MC));
        matrix[N_points[n], N_MC] K_y_MC = K(x1[sum(N_points[:n-1])+1:sum(N_points[:n])], x2, T1[sum(N_points[:n-1])+1:sum(N_points[:n])], T2, order);
        matrix[N_points[n], N_points[n]] cov_y = K_y[sum(N_points[:n-1])+1:sum(N_points[:n]), sum(N_points[:n-1])+1:sum(N_points[:n])] + diag_matrix(error*abs(y1[sum(N_points[:n-1])+1:sum(N_points[:n])])+v_D[n]) - mdivide_right_spd(K_y_MC, cov_MC) * K_y_MC';
        vector[N_points[n]] mean_y = mdivide_right_spd(K_y_MC, cov_MC) * y_MC[n,:]';

        y1[sum(N_points[:n-1])+1:sum(N_points[:n])] ~ multi_normal(mean_y, cov_y);
    }
}
