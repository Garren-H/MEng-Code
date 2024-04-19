
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
}

data {
    int N_points;
    vector<lower=0, upper=1>[N_points] x;
    vector[N_points] T;
    vector[N_points] y;
    vector[4] scaling;
    real<lower=0> a;
}

transformed data {
    real<lower=0> error=0.01;
}

parameters {
    vector[4] p12_raw;
    vector[4] p21_raw;
    real<lower=0> v;
}

model {
    vector[4] p12 = p12_raw .* scaling;
    vector[4] p21 = p21_raw .* scaling;
    vector[N_points] y_means = NRTL(x, T, p12, p21, a);
    
    p12_raw ~ std_normal();
    p21_raw ~ std_normal();

    v ~ exponential(2);

    y ~ normal(y_means, sqrt(square(error*abs(y))+v));
}
