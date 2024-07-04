'''
Function file to create a valid stan model such 
that the variances are in increasing order for
the Hybrid model
'''

def generate_stan_code(include_clusters=False, variance_known=False):
    # include_clusters: whether to include cluster data
    #                 : If true include number of cluster, cluster 
    #                   assignment as a matrix, and the within cluster varaince 
    #                   (vector for each cluster) as inputs to the model
    # variance_known: whether the data-model variance is known
    #               : If true include the data-model variance as input to the model

    model_code = '''
    functions {
        vector NRTL(vector x, vector T, vector p12, vector p21, real a, matrix map_tij, matrix map_tij_dT) {
            int N = rows(x);
            vector[N] t12 = map_tij * p12;
            vector[N] t21 = map_tij * p21;
            vector[N] dt12_dT = map_tij_dT * p12;
            vector[N] dt21_dT = map_tij_dT * p21;   
            vector[N] at12 = a * t12;
            vector[N] at21 = a * t21;
            vector[N] G12 = exp(-at12);
            vector[N] G21 = exp(-at21);
            vector[N] term1 = ( ( (1-x) .* G12 .* (1 - at12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
            vector[N] term2 = ( ( x .* G21 .* (1 - at21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
            return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2 );
        }'''

    if include_clusters:
        model_code += '''
        real ps_like(array[] int N_slice, int start, int end, vector y, vector x, vector T, array[] matrix U_raw, 
          array[] matrix V_raw, array[] matrix U_raw_means, array[] matrix V_raw_means, vector v_ARD, vector v, 
          vector scaling, real a, real error, array[] int N_points, array[,] int Idx_known, array[] matrix mapping, 
          vector var_data, matrix sigma_cluster, matrix C, int D) {
            real all_target = 0;
            for (i in start:end) {
                vector[4] p12_raw;
                vector[4] p21_raw;
                vector[N_points[i]] y_std = sqrt(var_data[sum(N_points[:i-1])+1:sum(N_points[:i])]+v[i]);
                vector[N_points[i]] y_means;

                for (j in 1:4) {
                    vector[D] ui = (U_raw[j,:,Idx_known[i,1]] .* sigma_cluster[:,Idx_known[i,1]] + U_raw_means[j] * C[:,Idx_known[i,1]]) .* v_ARD;
                    vector[D] vj = (V_raw[j,:,Idx_known[i,2]] .* sigma_cluster[:,Idx_known[i,2]] + V_raw_means[j] * C[:,Idx_known[i,2]]);
                    vector[D] uj = (U_raw[j,:,Idx_known[i,2]] .* sigma_cluster[:,Idx_known[i,2]] + U_raw_means[j] * C[:,Idx_known[i,2]]) .* v_ARD;
                    vector[D] vi = (V_raw[j,:,Idx_known[i,1]] .* sigma_cluster[:,Idx_known[i,1]] + V_raw_means[j] * C[:,Idx_known[i,1]]);
                    p12_raw[j] = dot_product(ui, vj);
                    p21_raw[j] = dot_product(uj, vi);
                }

                y_means = NRTL(x[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                T[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                p12_raw, p21_raw, a,
                                mapping[1][sum(N_points[:i-1])+1:sum(N_points[:i]),:],
                                mapping[2][sum(N_points[:i-1])+1:sum(N_points[:i]),:]);
                all_target += normal_lpdf(y[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_means, y_std);
            }
            return all_target;
        }'''

    else:
        model_code += '''
        real ps_like(array[] int N_slice, int start, int end, vector y, vector x, vector T, array[] matrix U_raw, 
          array[] matrix V_raw, vector v_ARD, vector v, vector scaling, real a, real error, array[] int N_points,
          array[,] int Idx_known, array[] matrix mapping, vector var_data) {
            real all_target = 0;
            for (i in start:end) {
                vector[4] p12_raw;
                vector[4] p21_raw;
                vector[N_points[i]] y_std = sqrt(var_data[sum(N_points[:i-1])+1:sum(N_points[:i])]+v[i]);
                vector[N_points[i]] y_means;

                for (j in 1:4) {
                    p12_raw[j] = dot_product(U_raw[j,:,Idx_known[i,1]] .* v_ARD, V_raw[j,:,Idx_known[i,2]]);
                    p21_raw[j] = dot_product(U_raw[j,:,Idx_known[i,2]] .* v_ARD, V_raw[j,:,Idx_known[i,1]]);
                }

                y_means = NRTL(x[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                T[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                p12_raw, p21_raw, a,
                                mapping[1][sum(N_points[:i-1])+1:sum(N_points[:i]),:],
                                mapping[2][sum(N_points[:i-1])+1:sum(N_points[:i]),:]);
                all_target += normal_lpdf(y[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_means, y_std);
            }
            return all_target;
        }'''

    model_code += '''
    }

    data {
        int<lower=1> N_known;                       // number of known data points
        array[N_known] int<lower=1> N_points;       // number of data points in each known data set
        vector[sum(N_points)] x;                    // mole fraction
        vector[sum(N_points)] T;                    // temperature
        vector[sum(N_points)] y;                    // excess enthalpy
        vector<lower=0>[4] scaling;                 // scaling factor for NRTL parameter
        real<lower=0> a;                            // alpha value for NRTL model
        int<lower=1> grainsize;                     // grainsize for parallelization
        int<lower=1> N;                             // number of compounds
        int<lower=1,upper=N> D;                     // number of features
        array[N_known,2] int<lower=1> Idx_known;    // indices of known data points
        real<lower=0>scale_upper;                   // upper bound for scale parameter'''
    if variance_known: # include known data-model variance data input
        model_code += '''
        vector<lower=0>[N_known] v;                 // known data-model variance'''

    if include_clusters: # include cluster data input
        model_code += '''   
        int<lower=1> K;                             // number of clusters
        matrix<lower=0, upper=1>[K, N] C;           // cluster assignment
        vector<lower=0>[K] v_cluster;               // within cluster variance'''

    model_code += '''
    }

    transformed data {
        real error = 0.01;                                                  // error in the data (fraction of experimental data)
        vector[sum(N_points)] var_data = square(error*y);                   // variance of the data
        array[2] matrix[sum(N_points),4] mapping;                           // temperature mapping
        array[N_known] int N_slice;                                         // slice indices for parallelization'''
    
    if include_clusters: # transform cluster variance parameters into a vector for easier processing
        model_code += '''
        matrix[D, N] sigma_cluster = rep_matrix(sqrt(v_cluster'), D) * C;   // within cluster stadard deviation matrix'''

    model_code += '''
        for (i in 1:N_known) {
            N_slice[i] = i;
        }

        mapping[1] = append_col(append_col(append_col(rep_vector(1.0, sum(N_points)), T),
                        1.0 ./ T), log(T));         // mapping for tij
        mapping[1] = mapping[1] .* rep_matrix(scaling', sum(N_points)); // scaling the mapping

        mapping[2] = append_col(append_col(append_col(rep_vector(0.0, sum(N_points)), rep_vector(1.0, sum(N_points))),
                        -1.0 ./ square(T)), 1.0 ./ T);    // mapping for dtij_dT
        mapping[2] = mapping[2] .* rep_matrix(scaling', sum(N_points)); // scaling the mapping
    }

    parameters {'''

    if not variance_known: # include data-model variance as parameter to model
        model_code += '''
        vector<lower=0>[N_known] v;                 // data-model variance'''

    if include_clusters: # include cluster means as parameters
        model_code += '''
        array[4] matrix[D,K] U_raw_means;           // U_raw cluster means
        array[4] matrix[D,K] V_raw_means;           // V_raw cluster means'''

    model_code += '''
        array[4] matrix[D,N] U_raw;                 // feature matrices U
        array[4] matrix[D,N] V_raw;                 // feature matrices V
        real<lower=0, upper=scale_upper> scale;     // scale dictating the strenght of ARD effect
        vector<lower=0>[D] sigma_ARD; // ARD standard deviations transformed to uniform scale
    }

    transformed parameters {
        vector[D] v_ARD = (scale * sigma_ARD) .^ 2; // effective ARD variances; sqrt(v_ARD) ~ half-cauchy(0, scale)
    }
    '''

    model_code += '''
    model {
        // Exponential prior on scale
        scale ~ exponential(5);

        // half-cauhcy prior for standard deviation of the feature matrices
        sigma_ARD ~ cauchy(0, 1);
    '''

    if not variance_known: # include data-model variance as parameter prior to model
        model_code += '''
        // Exponential prior for variance-model mismatch
        v ~ exponential(2);
        '''
        
    model_code += '''
        // Priors for feature matrices
        for (i in 1:4) {
            to_vector(U_raw[i]) ~ std_normal();
            to_vector(V_raw[i]) ~ std_normal();'''

    if include_clusters: # include cluster mean priors
        model_code += '''
            // Priors for cluster means
            to_vector(U_raw_means[i]) ~ std_normal();
            to_vector(V_raw_means[i]) ~ std_normal();'''
    model_code += '''
        }

        // Likelihood function'''
    if include_clusters:
        model_code += '''
        target += reduce_sum(ps_like, N_slice, grainsize, y, x, T, U_raw, 
                                V_raw, U_raw_means, V_raw_means, v_ARD, v, scaling, a, error, N_points,
                                Idx_known, mapping, var_data, sigma_cluster, C, D);
        '''
    else:
        model_code += '''
        target += reduce_sum(ps_like, N_slice, grainsize, y, x, T, U_raw, 
                                V_raw, v_ARD, v, scaling, a, error, N_points,
                                Idx_known, mapping, var_data);'''
    model_code += '''
    }

    generated quantities {
        vector[N_known] log_lik;
        for (i in 1:N_known) {'''

    if include_clusters:
        model_code += '''
            log_lik[i] = ps_like(N_slice, i, i, y, x, T, U_raw, V_raw, U_raw_means, V_raw_means, v_ARD, v, scaling, a, error, 
                                N_points, Idx_known, mapping, var_data, sigma_cluster, C, D);'''
    else:
        model_code += '''
            log_lik[i] = ps_like(N_slice, i, i, y, x, T, U_raw, 
                                V_raw, v_ARD, v, scaling, a, error, 
                                N_points, Idx_known, mapping, var_data);'''
    model_code += '''
        }
    }
    '''

    return model_code