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
        }

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
        }
    }


    data {
        int N_known;                    // number of known data points
        array[N_known] int N_points;    // number of data points in each known data set
        vector[sum(N_points)] x;        // mole fraction
        vector[sum(N_points)] T;        // temperature
        vector[sum(N_points)] y;        // excess enthalpy
        vector[4] scaling;              // scaling factor for NRTL parameter
        real a;                         // alpha value for NRTL model
        int grainsize;                  // grainsize for parallelization
        int N;                          // number of compounds
        int D;                          // number of features
        array[N_known,2] int Idx_known; // indices of known data points'''
    if variance_known: # include known data-model variance data input
        model_code += '''
        vector<lower=0>[N_known] v;     // known data-model variance'''

    if include_clusters: # include cluster data input
        model_code += '''   
        int K;                          // number of clusters
        matrix[K, N] C;                 // cluster assignment
        vector<lower=0>[K] v_cluster;   // within cluster variance'''

    model_code += '''
    }

    transformed data {
        real error = 0.01;                      // error in the data (fraction of experimental data)
        vector[sum(N_points)] var_data = square(error*y);    // variance of the data
        array[2] matrix[sum(N_points),4] mapping;           // temperature mapping
        array[N_known] int N_slice;             // slice indices for parallelization
    '''
    
    if include_clusters: # transform cluster variance parameters into a vector for easier processing
        model_code += '''
        vector[D*N] E_cluster = to_vector(rep_matrix(v_cluster', D) * C); // within cluster variance matrix-flattened to vector'''

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
        vector<lower=0>[N_known] v;       // data-model variance'''

    if include_clusters: # include cluster means as parameters
        model_code += '''
        array[4] matrix[D,K] U_raw_means; // U_raw cluster means
        array[4] matrix[D,K] V_raw_means; // V_raw cluster means'''

    model_code += '''
        array[4] matrix[D,N] U_raw;       // feature matrices U
        array[4] matrix[D,N] V_raw;       // feature matrices V
        real<lower=1> scale;              // scale dictating the strenght of ARD effect
        vector<lower=0>[D] v_ARD;         // ARD variances aranged in increasing order with lower bound zero
    }
    '''

    model_code += '''
    model {
        // Gamma Prior for scale
        profile("Scale Prior"){
            scale ~ gamma(1e-9, 1e-9);
        }

        // ARD Exponential prior
        profile("ARD Prior"){
            v_ARD ~ exponential(scale);
        }
    '''

    if not variance_known: # include data-model variance as parameter prior to model
        model_code += '''
        // Exponential prior for variance-model mismatch
        profile("Data-Model Mismatch Prior"){
            v ~ exponential(2);
        }
        '''

    if include_clusters: # include cluster mean priors
        model_code += '''
        // Priors for cluster means and feature matrices 
        profile("Cluster Mean Feature Matrices"){
            for (i in 1:4) {
                to_vector(U_raw_means[i]) ~ std_normal();
                to_vector(V_raw_means[i]) ~ std_normal();
                to_vector(U_raw[i]) ~ normal(to_vector(U_raw_means[i] * C), E_cluster);
                to_vector(V_raw[i]) ~ normal(to_vector(V_raw_means[i] * C), E_cluster);
            }
        }
        '''
    else: # exclude cluster parameters 
        model_code += '''
        // Priors for feature matrices
        profile("Feature Matrices"){
            for (i in 1:4) {
                to_vector(U_raw[i]) ~ std_normal();
                to_vector(V_raw[i]) ~ std_normal();
            }
        }
        '''
    model_code += '''
        // Likelihood function
        profile("Likelihood"){
            target += reduce_sum(ps_like, N_slice, grainsize, y, x, T, U_raw, 
                                    V_raw, v_ARD, v, scaling, a, error, N_points,
                                    Idx_known, mapping, var_data);
        }
    }
    '''

    return model_code