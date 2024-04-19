
functions {
    vector NRTL(vector x, real T, real t12, real t21, real dt12_dT, real dt21_dT, real a) {
        int N = rows(x);
        real G12 = exp(-a * t12);
        real G21 = exp(-a * t21);
        vector[N] term1 = ( ( (1-x) * G12 * (1 - a*t12) + x * square(G12) ) ./ square((1-x) + x * G12) ) * dt12_dT;
        vector[N] term2 = ( ( x * G21 * (1 - a*t21) + (1-x) * square(G21) ) ./ square(x + (1-x) * G21) ) * dt21_dT;
        return -8.314 * square(T) * x .* (1-x) .* ( term1 + term2);
    }
}

data {
    int N_points;
    vector<lower=0, upper=1>[N_points] x;
    vector[N_points] y;
    vector[4] scaling;
    real<lower=0> a;
    int N_temps;
    vector[N_temps] T_unique;
    array[N_temps, 2] int T_idx;
}

transformed data {
    real<lower=0> error=0.01;
    matrix[2*N_temps, 4] tau_base;
    matrix[2*N_temps, 2*N_temps] KNN;

    for (i in 1:N_temps) {
        tau_base[i, :] = [1, T_unique[i], 1.0 / T_unique[i], log(T_unique[i])];
        tau_base[i+N_temps, :] = [0, 1, -1.0 / square(T_unique[i]), 1.0 / T_unique[i]];
    }

    KNN = add_diag(tau_base * tau_base', 1e-8);
}

parameters {
    vector[N_temps] t12;
    vector[N_temps] t21;
    vector[N_temps] dt12_dT;
    vector[N_temps] dt21_dT;
    real<lower=0> v;
}

model {
    vector[2*N_temps] all12 = append_row(t12, dt12_dT);
    vector[2*N_temps] all21 = append_row(t21, dt21_dT);
    
    all12 ~ multi_normal(rep_vector(0, 2*N_temps), KNN);
    all21 ~ multi_normal(rep_vector(0, 2*N_temps), KNN);
    
    v ~ exponential(2);

    {
        vector[N_points] y_means;
        for (i in 1:N_temps) {
            y_means[T_idx[i,1]:T_idx[i,2]] = NRTL(x[T_idx[i,1]:T_idx[i,2]], T_unique[i], t12[i], t21[i], dt12_dT[i], dt21_dT[i], a);
        }
        y ~ normal(y_means, sqrt(square(error*abs(y))+v));
    }
}
