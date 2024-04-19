
functions {
    vector NRTL(vector x, vector T, vector p12, vector p21, real a) {
        int N = rows(x);
        vector[N] t12 = p12[1] + p12[2] * T + p12[3] ./ T + p12[4] * log(T);
        vector[N] t21 = p21[1] + p21[2] * T + p21[3] ./ T + p21[4] * log(T);
        vector[N] dt12_dT = p12[2] - p12[3] ./ square(T) + p12[4] ./ T;
        vector[N] dt21_dT = p21[2] - p21[3] ./ square(T) + p21[4] ./ T;   
        vector[N] G12 = exp(-a * t12);
        vector[N] G21 = exp(-a * t21);
        vector[N] term1 = ( ( (1-x) .* G12 .* (1 - a*t12) + x .* square(G12) ) ./ square((1-x) + x .* G12) ) .* dt12_dT;
        vector[N] term2 = ( ( x .* G21 .* (1 - a*t21) + (1-x) .* square(G21) ) ./ square(x + (1-x) .* G21) ) .* dt21_dT;
        return -8.314 * square(T) .* x .* (1-x) .* ( term1 + term2);
    }

    real ps_like(array[] int N_slice, int start, int end, vector y, vector x, vector T, array[] matrix U_raw, 
      array[] matrix V_raw, vector v_ARD, vector v, vector scaling, real a, real error, array[] int N_points,
      array[,] int Idx_known) {
        real all_target = 0;
        for (i in start:end) {
            vector[4] p12_raw;
            vector[4] p21_raw;
            vector[N_points[i]] y_std = sqrt(square(error*y[sum(N_points[:i-1])+1:sum(N_points[:i])])+v[i]);
            vector[N_points[i]] y_means;

            for (j in 1:4) {
                p12_raw[j] = dot_product(U_raw[j, :, Idx_known[i,1]].*v_ARD, V_raw[j, :, Idx_known[i,2]]);
                p21_raw[j] = dot_product(U_raw[j, :, Idx_known[i,2]].*v_ARD, V_raw[j, :, Idx_known[i,1]]);
            }

            y_means = NRTL(x[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                T[sum(N_points[:i-1])+1:sum(N_points[:i])], 
                                                p12_raw .* scaling, p21_raw .* scaling, a);
            all_target += normal_lpdf(y[sum(N_points[:i-1])+1:sum(N_points[:i])] | y_means, y_std);
        }
        return all_target;
      }
}

data {
    int N_known;
    array[N_known] int N_points;
    vector[sum(N_points)] x;
    vector[sum(N_points)] T;
    vector[sum(N_points)] y;
    vector[4] scaling;
    real a;
    int grainsize;

    int N;
    int D;
    array[N_known,2] int Idx_known;
}

transformed data {
    real error = 0.01;
    array[N_known] int N_slice;
    for (i in 1:N_known) {
        N_slice[i] = i;
    }
}

parameters {
    array[4] matrix[D,N] U_raw;
    array[4] matrix[D,N] V_raw;
    vector<lower=0>[D] v_ARD;
    real<lower=0, upper=5> scale;
    vector<lower=0>[N_known] v;
}

model {
    v_ARD ~ exponential(scale);
    for (i in 1:4) {
        to_vector(U_raw[i]) ~ std_normal();
        to_vector(V_raw[i]) ~ std_normal();
    }

    v ~ exponential(2);

    target += reduce_sum(ps_like, N_slice, grainsize, y, x, T, U_raw, 
                            V_raw, v_ARD, v, scaling, a, error, N_points,
                            Idx_known);
}
