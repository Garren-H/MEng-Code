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
            return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
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
                                p12_raw .* scaling, p21_raw .* scaling, a,
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
        vector[N] var_data = square(0.01*y);    // variance of the data
        array[2] matrix[N,4] mapping;           // temperature mapping
        array[N_known] int N_slice;             // slice indices for parallelization
    '''
    
    if include_clusters: # transform cluster variance parameters into a vector for easier processing
        model_code += '''
        matrix[D, N] E_cluster = rep_matrix(v_cluster', D) * C; // within cluster varaince matrix'''

    model_code += '''
        for (i in 1:N_known) {
            N_slice[i] = i;
        }

        mapping[1] = append_col(append_col(append_col(rep_vector(1.0, N), T),
                        1.0 ./ T), log(T));         // mapping for tij

        mapping[2] = append_col(append_col(append_col(rep_vector(0.0, N), rep_vector(1.0, N)),
                        -1.0 ./ square(T)), 1.0 ./ T);    // mapping for dtij_dT
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
        real<lower=0, upper=5> scale;     // scale dictating the strenght of ARD effect'''

    # generate different parameters for each ARD variance parameter, and lower bound by previous
    model_code += '''
        real<lower=0> v_ARD_1;            // ARD variance parameter 1'''
    for d in range(1,D):
        model_code += f'''
        real<lower=v_ARD_{d}> v_ARD_{d+1};      // ARD variance parameter {d+1}'''
    model_code += '''
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
        for (i in 1:4) {
            to_vector(U_raw_means[i]) ~ std_normal();
            to_vector(V_raw_means[i]) ~ std_normal();
            to_vector(U_raw[i]) ~ normal(to_vector(U_raw_means[i] * C), to_vector(E_cluster));
            to_vector(V_raw[i]) ~ normal(to_vector(V_raw_means[i] * C), to_vector(E_cluster));
        }
        '''
    else: # exclude cluster parameters 
        model_code += '''
        // priors for feature matrices
        for (i in 1:4) {
            to_vector(U_raw[i]) ~ std_normal();
            to_vector(V_raw[i]) ~ std_normal();
        }
        '''
    model_code += '''
        // Likelihood function
        target += reduce_sum(ps_like, N_slice, grainsize, y, x, T, U_raw, 
                                V_raw, v_ARD, v, scaling, a, error, N_points,
                                Idx_known, mapping, var_data);
    }
    '''

    return model_code