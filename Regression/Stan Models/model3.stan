// This stan code is for finding the posterior distribution when we have initialzations
// It may be used when we do not have any initializations but results may not be useful
// Without initializations, the variance is typically learned to be large
functions {
    vector NRTL(vector x, vector T, vector p12, vector p21, real a) { // function for the NRTL model
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
}

data {
    int N_points;                             // Number of datapoints
    vector<lower=0, upper=1>[N_points] x;     // vector of compositions
    vector[N_points] T;                       // vector of temperatures
    vector[N_points] y;                       // vector of experimental excess enthalpies
    vector[4] scaling;                        // scaling vector for each parameter a_ij, b_ij, c_ij, d_ij
    real<lower=0> a;                          // constant alpha parameter fixed to 0.3
}

transformed data {
    real<lower=0> error=0.01;                 // %experimental error, assumed to be 1% of data
}

parameters {
    vector[4] p12_raw;                        // scaled [a_ij, b_ij, c_ij, d_ij] = pij_raw .* scaling parameters  
    vector[4] p21_raw;
    real<lower=0> v;                          // data-model mismatch parameter
}

model {
    vector[4] p12 = p12_raw .* scaling;                    // pij = [a_ij, b_ij, c_ij, d_ij]
    vector[4] p21 = p21_raw .* scaling;
    vector[N_points] y_means = NRTL(x, T, p12, p21, a);    // compute the predicted values
    
    p12_raw ~ std_normal();        // priors on p12
    p21_raw ~ std_normal();        // priors on p21

    v ~ exponential(2);            // prior on v

    y ~ normal(y_means, sqrt(square(error*abs(y))+v));     // likelihood function
}
